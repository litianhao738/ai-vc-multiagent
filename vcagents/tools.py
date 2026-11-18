"""智能体可用的工具：联网搜索、网页抓取、上传资料检索。"""
from __future__ import annotations

import io
import ipaddress
import json
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .config import OPENROUTER_BASE_URL
from .llm import ToolCall, ToolResult, ToolSpec, governed_request, _vendor
from .governor import RunStopped
from .knowledge import KnowledgeStore
from .quantities import claim_quantities, fact_quantities, loose_text, same_value, unmatched as unmatched_quantities
from .grounding import USER_KEY, _cite_keys, _rounds_to
from .report_review import table_amount_options
from .citations import (still_missing, supporting_text, SourceRegistry, as_of, compose_claim, fact_issues, normal_text as _normal_text, resolve,
                        snapshot_ref, near_text)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
MAX_PAGE_CHARS = 10000
# 只拦截本机和常见内网网段；不用 is_private，避免误伤代理软件 fake-ip 模式下的 198.18.0.0/15
BLOCKED_NETWORKS = [
    ipaddress.ip_network(n)
    for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "::1/128", "fc00::/7")
]


class ToolError(Exception):
    pass


# ---------------------------------------------------------------- 上传资料

class DocumentStore:
    """用户上传的 BP、财务材料等，按页保存，支持关键词检索。"""

    page_chars = 3000

    def __init__(self) -> None:
        self.docs: dict[str, list[str]] = {}

    def __bool__(self) -> bool:
        return bool(self.docs)

    def add_file(self, name: str, data: bytes) -> None:
        if name.lower().endswith(".pdf"):
            from pypdf import PdfReader

            self.docs[name] = [(p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(data)).pages]
        else:
            self.add_text(name, data.decode("utf-8", errors="ignore"))

    def add_text(self, name: str, text: str) -> None:
        self.docs[name] = [text[i : i + self.page_chars] for i in range(0, len(text), self.page_chars)] or [""]

    def overview(self) -> str:
        lines = []
        for name, pages in self.docs.items():
            preview = re.sub(r"\s+", " ", " ".join(pages))[:200]
            lines.append(f"- 《{name}》共 {len(pages)} 页：{preview}…")
        return "\n".join(lines)

    def search(self, query: str, k: int = 6) -> str:
        terms = _terms(query)
        hits = []
        for name, pages in self.docs.items():
            for i, page in enumerate(pages, 1):
                low = page.lower()
                score = sum(low.count(t) for t in terms)
                if score:
                    hits.append((score, name, i, page, low))
        if not hits:
            return f"上传资料中没有找到与“{query}”相关的内容。"
        hits.sort(key=lambda h: -h[0])
        out = []
        for _, name, i, page, low in hits[:k]:
            start = max(0, min(low.find(t) for t in terms if t in low) - 300)
            out.append(f"《{name}》第 {i} 页：…{page[start : start + 800].strip()}…")
        return "\n\n".join(out)

    def read(self, name: str, start_page: int, end_page: int) -> str:
        pages = self.docs.get(name)
        if pages is None:
            raise ToolError(f"没有名为《{name}》的资料，可选：{'、'.join(self.docs)}")
        start = max(1, start_page)
        end = min(len(pages), max(start, end_page))
        return "\n\n".join(f"【第 {i} 页】\n{pages[i - 1]}" for i in range(start, end + 1))[: MAX_PAGE_CHARS * 2]


def _terms(query: str) -> list[str]:
    words = [w.lower() for w in re.split(r"[\s,，。;；、]+", query) if w]
    terms = set(words)
    for w in words:  # 中文没有空格，补充二元切分，让“融资历史”也能命中“融资”
        if re.search(r"[一-鿿]", w) and len(w) > 2:
            terms.update(w[i : i + 2] for i in range(len(w) - 1))
    return list(terms)


# ---------------------------------------------------------------- 联网

class SonarSearch:
    """通过 OpenRouter 调用 Perplexity Sonar 联网搜索，返回带来源链接的摘要。"""

    retries = 3
    timeout = 120

    def __init__(self, api_key: str, model: str = "perplexity/sonar", governor=None):
        import openai

        self.openai = openai
        self.client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, max_retries=self.retries,
                                    timeout=self.timeout)
        self.model = model
        self.provider = _vendor(model)
        self.governor = governor
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd: float | None = None
        self.cost_complete = True
        self._lock = threading.Lock()

    def search(self, query: str) -> str:
        resp = governed_request(self, self.openai, lambda client: client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"检索与下面问题相关的公开信息，列出关键事实和数字，每条注明来源。\n问题：{query}",
            }],
        ), retries=self.retries, timeout=self.timeout)
        with self._lock:
            self.calls += 1
            if resp.usage:
                self.input_tokens += resp.usage.prompt_tokens or 0
                self.output_tokens += resp.usage.completion_tokens or 0
                cost = (getattr(resp.usage, "model_extra", None) or {}).get("cost")
                if isinstance(cost, (int, float)) and self.governor is not None:
                    self.governor.add_cost(cost)
                if isinstance(cost, (int, float)) and self.cost_complete:
                    self.cost_usd = (self.cost_usd or 0.0) + cost
                else:
                    self.cost_complete = False
                    self.cost_usd = None
            else:
                self.cost_complete = False
                self.cost_usd = None
        msg = resp.choices[0].message
        text = (msg.content or "").strip() or f"没有搜索到与“{query}”相关的结果。"
        sources = _citations(resp, msg)
        if sources:
            text += "\n\n来源：\n" + "\n".join(f"[{i}] {title} {url}".replace("  ", " ") for i, (title, url) in enumerate(sources, 1))
        return text


def _citations(resp, msg) -> list[tuple[str, str]]:
    """OpenRouter 把来源放在 message.annotations 里；部分模型还会在响应顶层附 citations。"""
    titles: dict[str, str] = {}  # url → 标题，按出现顺序去重
    for item in getattr(msg, "annotations", None) or []:
        data = item if isinstance(item, dict) else item.model_dump()
        cite = data.get("url_citation") or {}
        if cite.get("url"):
            titles.setdefault(cite["url"], cite.get("title") or "")
    for url in (getattr(resp, "model_extra", None) or {}).get("citations") or []:
        if isinstance(url, str):
            titles.setdefault(url, "")
    return [(title, url) for url, title in titles.items()]


def fetch_url(url: str, *, knowledge: KnowledgeStore | None = None) -> str:
    _check_public_url(url)
    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=20)
    resp.raise_for_status()
    if "pdf" in resp.headers.get("content-type", "") or url.lower().endswith(".pdf"):
        from pypdf import PdfReader

        title = url
        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(resp.content)).pages)
    else:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else url
        text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()
    snapshot = knowledge.archive(str(resp.url), title, text) if knowledge is not None and text else None
    if len(text) < 50:
        text += "\n（页面正文很少，可能需要登录、验证或依赖 JavaScript 渲染。）"
    suffix = "\n…（已截断）" if len(text) > MAX_PAGE_CHARS else ""
    location = (f"\n快照：{snapshot['snapshot_id']}\n抓取时间：{snapshot['fetched_at']}"
                f"\nSHA256：{snapshot['content_sha256']}\n长文可用 read_source 分块阅读。" if snapshot else "")
    return f"标题：{title}\nURL：{resp.url}{location}\n\n{text[:MAX_PAGE_CHARS]}{suffix}"


def _check_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError("只支持 http/https 网址")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ToolError(f"无法解析域名：{parsed.hostname}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if any(ip in net for net in BLOCKED_NETWORKS):
            raise ToolError("出于安全考虑，不抓取本机或内网地址")


# ---------------------------------------------------------------- 工具箱

def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOL_SPECS = {
    "search_memory": ToolSpec(
        "search_memory", "查询当前赛道和研究对象的历史事实线索、判断及纠错。旧记忆不能直接作为证据，须重新读取所列原文。",
        _obj({"query": {"type": "string", "description": "要复用或避免重犯的问题"}}, ["query"]),
    ),
    "search_knowledge": ToolSpec(
        "search_knowledge",
        "检索本地已归档的公开来源原文，返回来源链接、抓取时间、快照和原文片段。用于定位证据，历史快照不能代表当前状态。",
        _obj({"query": {"type": "string", "description": "具体关键词，含公司名和事实对象"}}, ["query"]),
    ),
    "read_source": ToolSpec(
        "read_source",
        "读取已归档来源的一段原文，可离线复查。snapshot_id 和 chunk_id 从检索结果获得；引用原URL并保留快照信息。",
        _obj({"snapshot_id": {"type": "string"}, "chunk_id": {"type": "integer", "minimum": 0}}, ["snapshot_id", "chunk_id"]),
    ),
    "web_search": ToolSpec(
        "web_search",
        "联网搜索公开信息（新闻、融资报道、官网、招聘、竞品等），返回带来源链接的摘要。一次只问一个具体问题，中英文都可以。",
        _obj({"query": {"type": "string", "description": "要检索的具体问题或关键词"}}, ["query"]),
    ),
    "fetch_url": ToolSpec(
        "fetch_url",
        "抓取网页或在线 PDF 的正文，用于阅读搜索结果的原文。",
        _obj({"url": {"type": "string", "description": "完整网址"}}, ["url"]),
    ),
    "search_documents": ToolSpec(
        "search_documents",
        "在用户上传的资料（BP、财务材料等）中按关键词检索，返回命中的段落及页码。",
        _obj({"query": {"type": "string", "description": "关键词，可用空格分隔多个"}}, ["query"]),
    ),
    "read_document": ToolSpec(
        "read_document",
        "阅读上传资料的指定页。",
        _obj(
            {
                "name": {"type": "string", "description": "资料文件名"},
                "start_page": {"type": "integer", "description": "起始页（从 1 开始）"},
                "end_page": {"type": "integer", "description": "结束页（含）"},
            },
            ["name", "start_page", "end_page"],
        ),
    ),
}
DOC_TOOLS = ("search_documents", "read_document")
KNOWLEDGE_TOOLS = ("search_knowledge", "read_source")


class Toolbox:
    def __init__(self, docs: DocumentStore | None = None, search: SonarSearch | None = None, company_website: str = "",
                 knowledge: KnowledgeStore | None = None, shared_reads: list[dict] | None = None,
                 registry: SourceRegistry | None = None, materials: dict | None = None,
                  known_values: set[float] | None = None, aliases=(), memory=None, memory_sector: str = "", memory_entity: str = "",
                  network_slots=None, document_scope: str = "", governor=None, rejection_review=None, input_amounts=None):
        self.docs = docs or DocumentStore()
        self.memory = memory
        self.memory_sector = memory_sector
        self.memory_entity = memory_entity
        self.network_slots = network_slots
        # 本轮运行管控：总时限与取消、相同搜索和网页读取的复用、网页抓取并发
        self.governor = governor
        self.rejection_review = rejection_review
        self.document_scope = document_scope
        # 本轮共享的来源编号表；工具返回的原文按句编号，引用只填编号和句号
        self.registry = registry if registry is not None else SourceRegistry()
        # 核查员的被核查材料，用于确认 target 是材料原句
        self.materials = {key: _squash(_material_text(value)) for key, value in (materials or {}).items()}
        # 用户输入与代码计算结果中的数值：不是外部事实，target 中出现时不要求在引用原文里找到
        self.known_values = known_values or set()
        self.input_amounts = input_amounts
        self.aliases = tuple(a for a in aliases if a)  # 研究对象名称及别名
        self.search = search
        self.knowledge = knowledge
        # 本轮其他角色实际读到的网页原文（read_records 格式），只用于核对摘录，不提供给模型
        self.shared_reads = shared_reads or []
        self._knowledge_reads: list[dict] = []
        self.company_host = _source_host(company_website) if company_website else ""
        self._web_reads: list[tuple[set[str], str]] = []
        self._document_reads: list[tuple[str, int, str]] = []

    def specs(self, names: tuple[str, ...]) -> list[ToolSpec]:
        return [TOOL_SPECS[n] for n in names if (n not in DOC_TOOLS or self.docs)
                and (n not in KNOWLEDGE_TOOLS or self.knowledge is not None)
                and (n != "search_memory" or self.memory is not None)]

    def execute(self, call: ToolCall) -> ToolResult:
        if self.governor is not None:
            self.governor.check()
        if self.network_slots is not None and call.name in ("web_search", "fetch_url"):
            with self.network_slots:
                return self._execute(call)
        return self._execute(call)

    def _web_search(self, query: str) -> str:
        """同一轮里相同的问题只检索一次；并发的重复调用等待同一个结果。"""
        if self.governor is None:
            return self.search.search(query)
        key = ("web_search", re.sub(r"\s+", " ", query).strip().casefold())
        return self.governor.shared(key, lambda: self.search.search(query))

    def _fetch(self, url: str) -> str:
        """同一网址本轮只抓取一次；每个工具箱仍各自登记读取记录与引用编号。"""
        if self.governor is None:
            return fetch_url(url, knowledge=self.knowledge)

        def fetch():
            with self.governor.slot("web"):
                return fetch_url(url, knowledge=self.knowledge)
        return self.governor.shared(("fetch_url", url.split("#", 1)[0].strip()), fetch)

    def _execute(self, call: ToolCall) -> ToolResult:
        if call.input is None:
            return ToolResult(call.id, (call.error or "工具参数不是合法 JSON") + "，请重试。", True)
        args = call.input
        try:
            if call.name == "search_memory":
                if self.memory is None:
                    raise ToolError("未启用研究记忆")
                content = json.dumps(self.memory.recall(self.memory_sector, self.memory_entity,
                                                       _required(args, "query"), kinds=("fact", "correction")), ensure_ascii=False)
            elif call.name == "web_search":
                if self.search is None:
                    raise ToolError("未配置联网搜索（需要 OPENROUTER_API_KEY）")
                content = self._web_search(_required(args, "query"))
            elif call.name == "fetch_url":
                url = _required(args, "url")
                content = self._fetch(url)
                self._record_web_read(url, content)
                content = self._cite_web(url, content)
            elif call.name == "search_knowledge":
                if self.knowledge is None:
                    raise ToolError("未启用本地来源库")
                hits = self.knowledge.search(_required(args, "query"))
                for hit in hits:
                    self._record_source_read({**hit, "text": hit["excerpt"]})
                    source = self._cite_snapshot(hit)
                    hit["source_id"], hit["excerpt"] = source.id, source.render(hit["start_offset"], hit["end_offset"])
                content = json.dumps(hits, ensure_ascii=False) if hits else "本地来源库没有命中，请联网检索原文。"
            elif call.name == "read_source":
                if self.knowledge is None:
                    raise ToolError("未启用本地来源库")
                chunk_id = args.get("chunk_id")
                if not isinstance(chunk_id, int) or isinstance(chunk_id, bool) or chunk_id < 0:
                    raise ToolError("chunk_id 必须是非负整数")
                record = self.knowledge.read(_required(args, "snapshot_id"), chunk_id)
                if record is None:
                    raise ToolError("该来源快照或分块不存在，请先搜索来源库")
                self._record_source_read(record)
                source = self._cite_snapshot(record)
                content = json.dumps({**record, "source_id": source.id,
                                      "text": source.render(record["start_offset"], record["end_offset"])}, ensure_ascii=False)
            elif call.name == "search_documents":
                content = self.docs.search(_required(args, "query"))
                self._record_document_search(content)
                content += "\n\n（检索结果只用于定位；引用前请用 read_document 阅读对应页，获得带编号的句子。）"
            elif call.name == "read_document":
                name = _required(args, "name")
                content = self.docs.read(name, int(args.get("start_page", 1)), int(args.get("end_page", 1)))
                pages = list(re.finditer(r"(?:^|\n\n)【第 (\d+) 页】\n", content))
                blocks = []
                for i, page in enumerate(pages):
                    end = pages[i + 1].start() if i + 1 < len(pages) else len(content)
                    number, shown = int(page.group(1)), content[page.end():end]
                    self._document_reads.append((name, number, shown))
                    source = self.registry.register_page(name, number, self.docs.docs[name][number - 1], self.document_scope)
                    blocks.append(f"【第 {number} 页】来源编号：{source.id}\n{source.render(0, len(shown))}")
                content = "\n\n".join(blocks)
            else:
                return ToolResult(call.id, f"未知工具：{call.name}", True)
        except RunStopped:
            raise  # 整份任务已停止，不能当成一次工具失败交给模型继续
        except ToolError as e:
            return ToolResult(call.id, str(e), True)
        except httpx.HTTPStatusError as e:
            return ToolResult(call.id, f"请求失败：HTTP {e.response.status_code}", True)
        except Exception as e:  # 工具失败不应中断智能体，把错误交给模型自行调整
            return ToolResult(call.id, f"工具执行出错：{type(e).__name__}: {e}", True)
        return ToolResult(call.id, content)

    def _record_web_read(self, requested_url: str, content: str) -> None:
        header, separator, body = content.partition("\n\n")
        final_url = re.search(r"(?m)^URL[：:]\s*(\S+)", header)
        aliases = {requested_url}
        if final_url:
            aliases.add(final_url.group(1))
        if separator:
            body = body.removesuffix("\n…（已截断）")
            body = body.removesuffix("\n（页面正文很少，可能需要登录、验证或依赖 JavaScript 渲染。）")
            self._web_reads.append((aliases, body))
            snapshot = re.search(r"(?m)^快照：([^\n]+)", header)
            if snapshot and self.knowledge is not None:
                record = self.knowledge.read(snapshot.group(1))
                self._knowledge_reads.append({**record, "text": body, "start_offset": 0, "end_offset": len(body)})

    def _cite_web(self, requested_url: str, content: str) -> str:
        """把网页正文换成带句号的版本；有快照时按完整快照编号，保证分块阅读时句号不变。"""
        header, separator, body = content.partition("\n\n")
        if not separator:
            return content
        thin = "\n（页面正文很少，可能需要登录、验证或依赖 JavaScript 渲染。）"
        truncated = body.endswith("\n…（已截断）")
        clean = body.removesuffix("\n…（已截断）").removesuffix(thin)
        final_url = re.search(r"(?m)^URL[：:]\s*(\S+)", header)
        title = re.search(r"(?m)^标题[：:]\s*(.*)$", header)
        snapshot_id = re.search(r"(?m)^快照：([^\n]+)", header)
        snapshot = self.knowledge.read(snapshot_id.group(1).strip()) if snapshot_id and self.knowledge is not None else None
        url = final_url.group(1) if final_url else requested_url
        source = self.registry.register_web(url, title.group(1) if title else url, snapshot["text"] if snapshot else clean,
                                            snapshot, aliases={requested_url, url})
        note = (f"\n…（已截断；可用 read_source 阅读该快照后续分块，句号沿用 {source.id} 的编号）" if truncated
                else thin if body.endswith(thin) else "")
        return f"{header}\n来源编号：{source.id}（引用时只填写来源编号和句号）\n\n{source.render(0, len(clean))}{note}"

    def _cite_snapshot(self, record: dict):
        full = self.knowledge.read(record["snapshot_id"])
        return self.registry.register_web(record["url"], record.get("title", ""), full["text"], full, aliases={record["url"]})

    def _record_source_read(self, record: dict) -> None:
        self._knowledge_reads.append(record)
        self._web_reads.append(({record["url"]}, record["text"]))

    def _record_document_search(self, content: str) -> None:
        hits = list(re.finditer(r"(?:^|\n\n)《([^》]+)》第 (\d+) 页：…", content))
        for i, hit in enumerate(hits):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(content)
            text = content[hit.end():end].rstrip().removesuffix("…")
            self._document_reads.append((hit.group(1), int(hit.group(2)), text))

    def verify_evidence(self, data: dict) -> dict:
        """匹配实际返回的同源连续摘录，不判断主张是否被摘录支持。"""
        sources = data["sources"]
        labels: dict[str, list[int]] = {}
        for i, source in enumerate(sources):
            leading = re.match(r"\s*((?:\[[A-Za-z0-9_-]+\]\s*)+)", source)
            ids = re.findall(r"\[([^]]+)\]", leading.group(1)) if leading else [str(i + 1)]
            for source_id in ids:
                labels.setdefault(source_id, []).append(i)
        evidence = []
        for item in data["evidence"]:
            if "quote" in item:
                evidence.append(self._pointer_evidence(item, sources, labels))
                continue
            excerpt = _normal_text(item["excerpt"])
            ref = item["source_ref"]
            ref_labels = re.findall(r"\[([A-Za-z0-9_-]+)\]", ref)
            ref_urls = _source_urls(ref)
            ref_docs = _document_names(ref, self.docs.docs)
            if ref_labels:
                candidates = [labels[label][0] for label in ref_labels if len(labels.get(label, [])) == 1]
                if len(candidates) != len(ref_labels):
                    candidates = []
            else:
                candidates = list(range(len(sources))) if ref_urls or ref_docs else []
            texts = []
            located = False
            company_source = False
            page_not_verified = False
            for i in candidates:
                source = sources[i]
                urls = _source_urls(source)
                names = _document_names(source, self.docs.docs)
                if ref_urls and not any(self._same_url(a, b) for a in ref_urls for b in urls):
                    continue
                if ref_docs and not set(ref_docs).intersection(names):
                    continue
                located = True
                if self.company_host and any(_source_host(url) == self.company_host for url in urls):
                    company_source = True
                for aliases, text in self._web_reads:
                    if aliases.intersection(urls) and (not ref_urls or aliases.intersection(ref_urls)):
                        if _page_groups(source) or _page_groups(ref):
                            page_not_verified = True
                        else:
                            texts.append(text)
                for name, page, text in self._document_reads:
                    if (name in names and (not ref_docs or name in ref_docs)
                            and _page_allowed(page, source) and _page_allowed(page, ref)):
                        texts.append(text)
            matched = bool(excerpt) and any(excerpt in _normal_text(text) for text in texts)
            shared = []
            if excerpt and not matched and located and not page_not_verified:
                wanted = ref_urls or {url for i in candidates for url in _source_urls(sources[i])}
                shared = [read for read in self.shared_reads
                          if wanted.intersection(read.get("urls") or [read.get("url")]) and excerpt in _normal_text(read["text"])]
                matched = bool(shared)
            status = ("matched" if matched else "missing_excerpt" if not excerpt else
                      "source_not_found" if not located else "page_not_verified" if page_not_verified else
                      "source_not_read" if not texts else "excerpt_not_found")
            record = {**item, "excerpt_verified": matched, "excerpt_status": status,
                      "writing_review": {}, "equivalent_values": []}
            record["source_snapshots"] = [
                {k: read[k] for k in ("snapshot_id", "url", "fetched_at", "content_sha256", "start_offset", "end_offset")}
                for read in self._knowledge_reads
                if matched and any(self._same_url(read["url"], url) for url in _source_urls(ref + " " + " ".join(sources[i] for i in candidates)))
                and excerpt in _normal_text(read["text"])
            ] + [{k: read[k] for k in ("snapshot_id", "url", "fetched_at", "content_sha256", "start_offset", "end_offset")}
                 for read in shared if read.get("kind") == "snapshot"]
            if shared:
                record["matched_via"] = "本轮其他角色已读取的同一来源"
            if company_source:
                record["source_type"] = "公司口径"
                record["source_type_note"] = "引用来源属于用户提供的公司官网，不作为独立来源。"
            evidence.append(record)
        return {**data, "evidence": evidence}

    def _quotes(self, fact: dict):
        quote = resolve(self.registry, fact.get("quote"))
        dq = fact.get("date_quote") or {}
        return quote, (resolve(self.registry, dq) if str(dq.get("source") or "").strip() else None)

    def _review_writing(self, issues, fact, quote, target=""):
        # Missing sources, invalid pointers, dates, required fields and length limits
        # never enter this review. Only compare writing against a valid read quote.
        eligible = [issue for issue in issues if issue.startswith(("数量", "名称", "statement 中的", "target 中的"))]
        if not eligible or not quote.ok or self.rejection_review is None:
            return issues
        remaining = self.rejection_review(eligible, {
            "submitted": {key: fact.get(key) for key in ("subject", "statement", "quantities", "entities", "date")},
            "target": target, "source_excerpt": near_text(quote),
            "source_url": quote.source.url, "source_location": quote.location})
        return [issue for issue in issues if issue not in eligible or issue in remaining]

    def _pointer_evidence(self, item: dict, sources: list[str], labels: dict[str, list[int]]) -> dict:
        """引用定位格式：摘录由代码按句号取出，各字段须在引用原文中逐字或等值出现。"""
        quote, date_quote = self._quotes(item)
        issues = fact_issues(item, quote, date_quote, self.aliases)
        original_issues = list(issues)
        status = quote.status
        if quote.ok:
            refs = re.findall(r"\[([A-Za-z0-9_-]+)\]", item.get("source_ref", ""))
            entries = [sources[labels[ref][0]] for ref in refs if len(labels.get(ref, [])) == 1]
            if not any(self._entry_is_source(entry, quote.source) for entry in entries):
                status = "source_mismatch"
                issues.append(f"source_ref {item.get('source_ref') or '（空）'} 对应的来源与引用的 {quote.source.id}（{quote.source.url}）不一致")
            else:
                issues = self._review_writing(issues, item, quote)
                if issues:
                    status = "fields_unsupported"
        verified = quote.ok and not issues
        waived = [issue for issue in original_issues if issue not in issues] if verified else []
        equivalent_text = item.get("statement", "") + " " + " ".join(q.get("text", "") for q in item.get("quantities", []))
        record = {**item, "claim": compose_claim(item), "excerpt": supporting_text(item, quote), "as_of": as_of(item),
                  "quote_location": quote.location, "quote_error": quote.error, "field_issues": issues,
                  "excerpt_verified": verified, "excerpt_status": status,
                  "source_snapshots": snapshot_ref(quote) if verified else [],
                  "writing_review": {"waived": waived} if waived else {},
                  "equivalent_values": sorted({q.value for q in fact_quantities(equivalent_text)}) if waived else []}
        if quote.ok and self.company_host and _source_host(quote.source.url) == self.company_host:
            record["source_type"] = "公司口径"
            record["source_type_note"] = "引用来源属于用户提供的公司官网，不作为独立来源。"
        return record

    def _entry_is_source(self, entry: str, source) -> bool:
        if "document" in source.meta:
            accepted = {source.meta.get("qualified_document", source.meta["document"])}
            if not source.meta.get("scope") or source.meta["scope"] == self.document_scope:
                accepted.add(source.meta["document"])
            return bool(accepted.intersection(_document_names(entry, self.docs.docs))) and _page_allowed(source.meta["page"], entry)
        return any(url in source.aliases or self._same_url(url, source.url) for url in _source_urls(entry))

    def _input_exemptions(self, target, source_agent, missing):
        """An input may explain a number missing from the quote only within its own scope."""
        if not missing or USER_KEY not in _cite_keys(target):
            return missing, set()
        quantities = claim_quantities(target)
        company = self.input_amounts.material_company(source_agent, "") if self.input_amounts else ""
        unresolved, protected = [], set()
        for q in quantities:
            if q.text not in missing:
                continue
            scope = self.input_amounts.check(target, q, quantities, company) if self.input_amounts else None
            if scope:
                if scope[0] == "supported":
                    continue
                protected.add(q.text)
            elif any(same_value(q.value, v) or _rounds_to(q.value, v) for v in self.known_values):
                continue
            unresolved.append(q.text)
        return unresolved, protected

    def _pointer_fact_check(self, claim: dict, company_hosts: set[str]) -> dict:
        fact = claim["fact"]
        quote, date_quote = self._quotes(fact)
        issues = fact_issues(fact, quote, date_quote, self.aliases)
        context = [quote.text] + ([date_quote.text] if date_quote and date_quote.ok
                                  and (fact.get("date") or {}).get("basis") == "来源其他位置" else [])
        target = claim.get("target", "")
        missing = still_missing(unmatched_quantities(target, *context), quote) if quote.ok else []
        missing, protected = self._input_exemptions(target, claim.get("source_agent", ""), missing)
        if quote.ok and claim.get("support") == "完全支持" and claim["verdict"] in ("属实", "公司口径"):
            # The auditor affirms the row's meaning; the program checks the explicit table unit.
            if missing and table_amount_options(target, quote.text):
                amount = fact_quantities(target)[0].text
                missing = [number for number in missing if number != amount]
            target_issues = {number: f"target 中的 {number} 不在引用原文中" for number in missing if number not in protected}
            remaining = self._review_writing(issues + list(target_issues.values()), fact, quote, target)
            issues = [issue for issue in issues if issue in remaining]
            missing = [number for number in missing if number in protected or target_issues[number] in remaining]
        verdict, status = claim["verdict"], quote.status
        if verdict in ("属实", "公司口径", "不实") and not quote.ok:
            verdict = "无法核实"
        elif verdict in ("属实", "公司口径") and (claim.get("support") != "完全支持" or missing):
            # 只有数字对不上或支持不完全才降级；名称、数量写法与原文略有差异只记录在 field_issues，不改变结论
            verdict = "无法核实"
            status = "fields_unsupported" if missing else "partial_support"
        elif verdict == "属实" and _source_host(quote.source.url) in company_hosts:
            verdict = "公司口径"
        correction = claim["correction"] if verdict == "不实" else ""
        if quote.ok and unmatched_quantities(correction, *context):
            correction = ""
        return {**claim, "claim": target or compose_claim(fact), "fact_claim": compose_claim(fact),
                "verification_source": quote.source.url if quote.source else "", "verification_excerpt": quote.text,
                "model_verdict": claim["verdict"], "verdict": verdict, "model_correction": claim["correction"],
                "correction": correction, "excerpt_verified": quote.ok, "verification_status": status,
                "quote_location": quote.location, "field_issues": issues, "unmatched_numbers": missing,
                "source_snapshots": snapshot_ref(quote)}

    def citation_errors(self, role_key: str, data: dict) -> list[str]:
        """提交时检查引用；修正次数由 Agent 统一限制。"""
        errors = []
        if role_key in ("sector_researcher", "company_researcher"):
            for i, item in enumerate(self.verify_evidence(data)["evidence"]):
                if "quote" in item and item["excerpt_status"] != "missing_quote" and not item["excerpt_verified"]:
                    errors.append(f"evidence[{i}]：" + "；".join(([item["quote_error"]] if item["quote_error"] else [])
                                                               + item["field_issues"]))
        elif role_key == "fact_checker":
            for i, claim in enumerate(data.get("claims", [])):
                if "fact" in claim:
                    errors += [f"claims[{i}]：{e}" for e in self._fact_check_errors(claim)]
        if errors:
            more = len(errors) - 10
            errors = errors[:10] + ([f"另有 {more} 处同类问题"] if more > 0 else [])
        return errors

    def _fact_check_errors(self, claim: dict) -> list[str]:
        errors = []
        material = self.materials.get(claim.get("source_agent", ""))
        if material is not None and not _close_to_material(_squash(claim.get("target", "")), material):
            errors.append(f"target 必须从材料 {claim.get('source_agent')} 中摘录原句（允许轻微改写）")
        if claim["verdict"] not in ("属实", "公司口径", "不实"):
            return errors
        checked = self._pointer_fact_check(claim, set())
        quote, _ = self._quotes(claim["fact"])
        if not quote.ok:
            errors.append(f"判“{claim['verdict']}”需要有效引用：{quote.error}")
            return errors
        if claim["verdict"] in ("属实", "公司口径") and claim.get("support") != "完全支持":
            errors.append("support 不是“完全支持”时不能判属实或公司口径")
        errors += checked["field_issues"]
        if claim["verdict"] in ("属实", "公司口径") and checked["unmatched_numbers"]:
            errors.append(f"target 中的 {'、'.join(checked['unmatched_numbers'])} 不在引用原文中：把 target 缩小到引用能直接证明的部分，"
                          "或拆成多条，或改判无法核实")
        return errors

    def _same_url(self, first: str, second: str) -> bool:
        return first == second or any(first in aliases and second in aliases for aliases, _ in self._web_reads)

    def verify_fact_check(self, data: dict, company_websites=()) -> dict:
        """核查结论须有本次读取的同源摘录；保留模型原结论供审阅。"""
        company_hosts = {_source_host(website) for website in company_websites if website} - {""}
        claims = []
        for claim in data["claims"]:
            if "fact" in claim:
                claims.append(self._pointer_fact_check(claim, company_hosts))
                continue
            source = claim["verification_source"]
            evidence = self.verify_evidence({
                "sources": [source],
                "evidence": [{"source_ref": source, "excerpt": claim["verification_excerpt"], "source_type": "未知"}],
            })["evidence"][0]
            verdict = claim["verdict"]
            if verdict in ("属实", "公司口径", "不实") and not evidence["excerpt_verified"]:
                verdict = "无法核实"
            elif verdict == "属实" and any(_source_host(url) in company_hosts for url in _source_urls(source)):
                verdict = "公司口径"
            missing_numbers = []
            if verdict in ("属实", "公司口径"):
                missing_numbers = unmatched_quantities(claim["claim"], claim["verification_excerpt"])
                if missing_numbers:
                    verdict = "无法核实"
            correction = claim["correction"] if verdict == "不实" else ""
            if unmatched_quantities(correction, claim["verification_excerpt"]):
                correction = ""
            claims.append({**claim, "model_verdict": claim["verdict"], "verdict": verdict,
                           "model_correction": claim["correction"], "correction": correction,
                           "excerpt_verified": evidence["excerpt_verified"],
                           "source_snapshots": evidence["source_snapshots"],
                           "unmatched_numbers": missing_numbers,
                           "verification_status": evidence["excerpt_status"]})
        return {**data, "claims": claims}

    def read_records(self) -> list[dict]:
        return ([{"kind": "snapshot", **record} for record in self._knowledge_reads]
                + [{"kind": "web", "urls": sorted(aliases), "text": text} for aliases, text in self._web_reads]
                + [{"kind": "document", "name": name, "page": page, "text": text}
                   for name, page, text in self._document_reads])

    def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        if len(calls) <= 1:
            return [self.execute(c) for c in calls]
        with ThreadPoolExecutor(max_workers=min(4, len(calls))) as pool:
            return list(pool.map(self.execute, calls))


def describe_call(call: ToolCall) -> str:
    args = call.input or {}
    return {
        "web_search": f"搜索：{args.get('query', '')}",
        "fetch_url": f"读取网页：{args.get('url', '')}",
        "search_knowledge": f"检索来源库：{args.get('query', '')}",
        "read_source": f"回读来源快照：{args.get('snapshot_id', '')[:12]} · 分块 {args.get('chunk_id', '')}",
        "search_documents": f"检索资料：{args.get('query', '')}",
        "read_document": f"阅读《{args.get('name', '')}》第 {args.get('start_page')}-{args.get('end_page')} 页",
    }.get(call.name, call.name)


def _required(args: dict, key: str) -> str:
    value = str(args.get(key, "")).strip()
    if not value:
        raise ToolError(f"缺少参数 {key}")
    return value


def normalize_fact_check(data: dict) -> dict:
    """演示模式不做原文校验，只把引用定位格式补成下游使用的字段。"""
    claims = []
    for claim in data.get("claims", []):
        if "fact" in claim:
            claim = {"verification_source": "", "verification_excerpt": "", **claim,
                     "claim": claim.get("target") or compose_claim(claim["fact"])}
        claims.append(claim)
    return {**data, "claims": claims}


def _material_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_material_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(_material_text(v) for v in value.values())
    return ""


def _close_to_material(target: str, material: str) -> bool:
    """target 是材料原句，或至少七成文字能在材料中按顺序找到（允许标点、措辞的轻微改写）。"""
    if not target or target in material:
        return True
    from difflib import SequenceMatcher

    blocks = SequenceMatcher(None, target, material, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= 4) >= 0.7 * len(target)


def _squash(text: str) -> str:
    """比较 target 与材料原句：忽略大小写、全半角、空白和标点。"""
    return loose_text(text)


def _source_urls(text: str) -> set[str]:
    return {url.rstrip(".,，。;；\"'）") for url in re.findall(r"https?://[^\s<>\]\)]+", text, re.IGNORECASE)}


def _source_host(url: str) -> str:
    try:
        host = urlparse(url if "://" in url else "https://" + url).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


def _document_names(text: str, known_names) -> list[str]:
    names = re.findall(r"《([^》]+)》", text)
    if not names:
        names = [name for name in known_names
                 if re.search(r"(?<![\w./\\-])" + re.escape(name) + r"(?![\w./\\-])", text)]
    return names


def _page_groups(reference: str) -> list[str]:
    groups = re.findall(r"第\s*([\d\s、,，\-–—~至到]+)\s*页", reference)
    groups += re.findall(r"\bpages?\s+([\d\s,\-–—~]+)", reference, re.IGNORECASE)
    groups += re.findall(r"\bpp?\.\s*([\d\s,\-–—~]+)", reference, re.IGNORECASE)
    return groups


def _page_allowed(page: int, reference: str) -> bool:
    groups = _page_groups(reference)
    if not groups:
        return True
    for group in groups:
        for part in re.split(r"[、,，]", group):
            numbers = [int(n) for n in re.findall(r"\d+", part)]
            if len(numbers) == 1 and page == numbers[0]:
                return True
            if len(numbers) == 2 and numbers[0] <= page <= numbers[1]:
                return True
    return False
