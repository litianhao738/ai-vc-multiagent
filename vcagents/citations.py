"""引用定位：原文按句编号，模型只提交“来源编号 + 句号”，摘录由代码取出。

同一来源在一次运行内只有一个编号（W 网页或来源库快照，D 上传资料页），所有角色共享。
切句只依赖来源文本；快照内容由 SHA256 固定，所以同一快照的句号永远不变。
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field

from .quantities import claim_quantities, loose_text, same_value, source_values, unmatched

SEGMENTER_VERSION = 1
MAX_SEGMENT = 200
MAX_QUOTE_SENTENCES = 5
MAX_FACT_STATEMENT_CHARS = 160
_BREAK = re.compile(r"(?<=[。！？；!?;])|(?<=\.)(?=\s+[A-Z\"“(])")
_SOFT_BREAK = re.compile(r"(?<=[，,、:：])")


_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"', "＂": '"', "‘": "'", "’": "'", "‚": "'", "‛": "'", "＇": "'"})
_CJK = r"⺀-鿿　-〿＀-￯"


def normal_text(text: str) -> str:
    """只统一排版差异：空白、引号样式，以及网页正文在中文与全角标点旁插入的空格。"""
    text = re.sub(r"\s+", " ", text.translate(_QUOTES)).strip()
    return re.sub(rf"(?<=[{_CJK}]) | (?=[{_CJK}])", "", text)


def segment_spans(text: str) -> list[tuple[int, int]]:
    """按换行和句末标点切句，超长句在逗号处或按长度再切；返回去除首尾空白后的字符区间。"""
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        cuts = [line.start()] + [line.start() + m.start() for m in _BREAK.finditer(line.group(0))] + [line.end()]
        for start, end in zip(cuts, cuts[1:]):
            spans.extend(_split_long(text, start, end))
    out = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            out.append((start, end))
    return out


def _split_long(text: str, start: int, end: int) -> list[tuple[int, int]]:
    if end - start <= MAX_SEGMENT:
        return [(start, end)]
    pieces, cursor = [], start
    while end - cursor > MAX_SEGMENT:
        window = text[cursor:cursor + MAX_SEGMENT]
        soft = [m.start() for m in _SOFT_BREAK.finditer(window) if m.start() > MAX_SEGMENT // 3]
        cut = cursor + (soft[-1] if soft else MAX_SEGMENT)
        pieces.append((cursor, cut))
        cursor = cut
    pieces.append((cursor, end))
    return pieces


@dataclass
class Source:
    id: str
    url: str  # 网页 URL；上传资料为“《文件名》第 n 页”
    title: str
    text: str
    spans: list[tuple[int, int]]
    meta: dict = field(default_factory=dict)  # snapshot_id、fetched_at、content_sha256 或 document、page
    aliases: set[str] = field(default_factory=set)
    shown: set[int] = field(default_factory=set)  # 本轮已返回给任一角色的句号（从 1 开始）

    def render(self, start: int = 0, end: int | None = None) -> str:
        """返回 [start, end) 字符范围内完整出现的句子，带编号，并记为已读。"""
        end = len(self.text) if end is None else end
        lines = []
        for i, (a, b) in enumerate(self.spans, 1):
            if a >= start and b <= end:
                self.shown.add(i)
                lines.append(f"[{self.id}:s{i}] {self.text[a:b]}")
        return "\n".join(lines)

    def location(self, first: int, last: int) -> str:
        return f"{self.id}:s{first}" if first == last else f"{self.id}:s{first}–s{last}"


class SourceRegistry:
    """一次运行内的来源编号表，线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: dict[str, Source] = {}
        self._keys: dict[str, str] = {}
        self._counts = {"W": 0, "D": 0}

    def register(self, kind: str, key: str, url: str, title: str, text: str, meta: dict | None = None,
                 aliases=()) -> Source:
        with self._lock:
            if key in self._keys:
                source = self._sources[self._keys[key]]
                source.aliases.update(a for a in aliases if a)
                return source
            self._counts[kind] += 1
            source = Source(f"{kind}{self._counts[kind]}", url, title, text, segment_spans(text), dict(meta or {}),
                            {url, *(a for a in aliases if a)})
            self._sources[source.id] = source
            self._keys[key] = source.id
            return source

    def register_web(self, url: str, title: str, text: str, snapshot: dict | None = None, aliases=()) -> Source:
        key = (f"snapshot:{snapshot['snapshot_id']}" if snapshot
               else "web:" + url + ":" + hashlib.sha256(text.encode("utf-8")).hexdigest())
        meta = {k: snapshot[k] for k in ("snapshot_id", "fetched_at", "content_sha256")} if snapshot else {}
        return self.register("W", key, url, title, text, meta, aliases)

    def register_page(self, document: str, page: int, text: str, scope: str = "") -> Source:
        qualified = f"{scope}/{document}" if scope else document
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.register("D", f"doc:{qualified}:{page}:{digest}", f"《{qualified}》第 {page} 页", qualified, text,
                             {"document": document, "qualified_document": qualified, "scope": scope,
                              "page": page, "content_sha256": digest})

    def get(self, source_id: str) -> Source | None:
        return self._sources.get(source_id.strip())

    def index(self) -> dict[str, dict]:
        """供结果文件保存：编号 → 来源元数据（不含正文）。"""
        return {s.id: {"url": s.url, "title": s.title, "segments": len(s.spans), "segmenter": SEGMENTER_VERSION,
                       **s.meta} for s in self._sources.values()}


@dataclass
class Quote:
    ok: bool
    status: str  # matched / missing_quote / source_not_found / quote_out_of_range / quote_too_long / source_not_read
    error: str = ""
    source: Source | None = None
    text: str = ""
    start: int = 0
    end: int = 0
    location: str = ""


def resolve(registry: SourceRegistry, quote: dict | None) -> Quote:
    quote = quote or {}
    source_id = str(quote.get("source") or "").strip()
    if not source_id:
        return Quote(False, "missing_quote", "没有给出引用位置")
    source = registry.get(source_id)
    if source is None:
        return Quote(False, "source_not_found", f"来源编号 {source_id} 不存在，只能引用工具返回过的编号")
    first, last = quote.get("from"), quote.get("to")
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in (first, last)) or not 1 <= first <= last <= len(source.spans):
        return Quote(False, "quote_out_of_range", f"{source_id} 的句号范围无效，应在 s1–s{len(source.spans)} 之间且 from ≤ to")
    if last - first + 1 > MAX_QUOTE_SENTENCES:
        return Quote(False, "quote_too_long", f"{source_id} 一次最多引用连续 {MAX_QUOTE_SENTENCES} 句")
    unread = [i for i in range(first, last + 1) if i not in source.shown]
    if unread:
        shown = sorted(source.shown)
        hint = f"已读范围 s{shown[0]}–s{shown[-1]}（可能不连续）" if shown else "尚未读取正文"
        return Quote(False, "source_not_read", f"{source_id}:s{unread[0]} 本轮未读到，{hint}；请先用工具阅读对应部分")
    start, end = source.spans[first - 1][0], source.spans[last - 1][1]
    return Quote(True, "matched", "", source, source.text[start:end], start, end, source.location(first, last))


_DATE_LIKE = re.compile(r"(?:19|20)\d{2}|\d{1,2}")


def near_text(quote: Quote, around: int = 2) -> str:
    """引用句及前后两句。模型常少圈一句，或主语、金额写在相邻句。"""
    spans = quote.source.spans
    touched = [i for i, (a, b) in enumerate(spans) if a < quote.end and b > quote.start]
    if not touched:
        return quote.text
    first, last = max(0, touched[0] - around), min(len(spans) - 1, touched[-1] + around)
    return quote.source.text[spans[first][0]:spans[last][1]]


def _document_context(source: Source) -> list[str]:
    """整篇来源、标题和网址里的日期，例如 /2026/10/26/ 或 20261026。"""
    dates = " ".join(f"{y}-{m}-{d}" for y, m, d in re.findall(r"(20\d{2})[-/]?(\d{2})[-/]?(\d{2})", source.url))
    return [source.text, source.title, dates]


def still_missing(numbers: list[str], quote: Quote) -> list[str]:
    """引用句里没有的数字：相邻句有就算；年月日只要整篇来源写过也算。其他数字仍须在引用附近。"""
    if not numbers or not quote.ok or quote.source is None:
        return numbers
    near = near_text(quote)
    out = []
    for text in numbers:
        if not unmatched(text, near):
            continue
        if _DATE_LIKE.fullmatch(text.strip()) and not unmatched(text, *_document_context(quote.source)):
            continue
        out.append(text)
    return out


def _quantity_found(text: str, quote: Quote) -> bool:
    """写法不同但数值相同就算找到，例如 $400 million 与 $400M、超过3000% 与 超过 3000%。"""
    near = near_text(quote)
    wanted = [q.value for q in claim_quantities(text)] or list(source_values(text))
    # Names may ignore punctuation. Quantities must retain decimal points and
    # signs: removing either can turn a formatting tolerance into a false fact.
    if not wanted:
        return bool(loose_text(text)) and loose_text(text) in loose_text(near)
    available = source_values(near)
    return all(any(same_value(v, s) for s in available) for v in wanted)


def fact_issues(fact: dict, quote: Quote, date_quote: Quote | None, aliases=()) -> list[str]:
    """检查结构化事实能否在引用原文中找到。只拦内容问题：大小写、标点、全半角、数字写法和日期出处不算错。
    主体和研究对象本身的名称（含别名，如 Liblib 与演语科技）不要求出现在引用句中，与 subject 一致。"""
    if not quote.ok:
        return []
    issues = []
    exempt = {loose_text(a) for a in (*aliases, fact.get("subject", "")) if a}
    statement = fact.get("statement", "")
    if len(statement) > MAX_FACT_STATEMENT_CHARS:
        issues.append(f"statement 超过 {MAX_FACT_STATEMENT_CHARS} 字，请拆成多条")
    for item in fact.get("quantities") or []:
        text = str(item.get("text", ""))
        if text.strip() and not _quantity_found(text, quote):
            issues.append(f"数量“{text}”在引用原文及相邻句中找不到")
    # 投资方等名称常写在相邻句或标题里；整篇来源都没有才算问题
    document = loose_text(" ".join((quote.source.text, quote.source.title, quote.source.url))) if quote.source else ""
    for name in fact.get("entities") or []:
        key = loose_text(name)
        if key and key not in document and not any(key in a or a in key for a in exempt):
            issues.append(f"名称“{name}”在该来源中找不到")
    date_text = str((fact.get("date") or {}).get("text", "")).strip()
    context = [quote.text] + ([date_quote.text] if date_quote is not None and date_quote.ok else [])
    if date_text and (missing := still_missing(unmatched(date_text, *context), quote)):
        issues.append(f"日期中的 {'、'.join(missing)} 在该来源中找不到；来源没写的日期不要补")
    if missing := still_missing(unmatched(statement, *context), quote):
        issues.append(f"statement 中的 {'、'.join(missing)} 在引用原文及相邻句中找不到等值写法")
    return issues


def supporting_text(fact: dict, quote: Quote) -> str:
    """数字只出现在相邻句时，摘录带上相邻句，下游按摘录核对数字才对得上。"""
    if not quote.ok:
        return quote.text
    wanted = [str(item.get("text", "")) for item in fact.get("quantities") or []] + [fact.get("statement", "")]
    return near_text(quote) if any(unmatched(text, quote.text) for text in wanted if text) else quote.text


def compose_claim(fact: dict) -> str:
    """由结构化字段拼出主张句，拼法固定，不会补出原文没有的内容。"""
    date = fact.get("date") or {}
    date_text = str(date.get("text", "")).strip()
    claim = f"{fact.get('subject', '').strip()}{('于' + date_text) if date_text else ''}{fact.get('statement', '').strip()}"
    missing = [e for e in fact.get("entities") or [] if e and e not in claim]
    if missing:
        claim += "（" + "、".join(missing) + "）"
    if date_text and date.get("basis") == "来源其他位置":
        claim += "（日期据来源其他位置推断）"
    return claim


def as_of(fact: dict) -> str:
    date = fact.get("date") or {}
    text = str(date.get("text", "")).strip()
    if not text:
        return ""
    return text + ("（推断）" if date.get("basis") == "来源其他位置" else "")


def snapshot_ref(quote: Quote) -> list[dict]:
    """引用的快照与字符位置；上传资料没有快照时返回空列表。"""
    source = quote.source
    if not quote.ok or source is None or "snapshot_id" not in source.meta:
        return []
    return [{"snapshot_id": source.meta["snapshot_id"], "url": source.url, "fetched_at": source.meta["fetched_at"],
             "content_sha256": source.meta["content_sha256"], "start_offset": quote.start, "end_offset": quote.end}]
