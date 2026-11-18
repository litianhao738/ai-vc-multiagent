"""Readable A4 report export with an actual, hard page-count limit."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from threading import Lock
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.doctemplate import LayoutError

from .config import DEFAULT_REPORT_MAX_PAGES, REPORT_PAGE_CAP

_FONT_LOCK = Lock()


class ReportPageLimitError(ValueError):
    pass


def _font():
    with _FONT_LOCK:
        if "ResearchCN" not in pdfmetrics.getRegisteredFontNames():
            candidates = [Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/simsun.ttc"),
                          Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")]
            for path in candidates:
                if path.exists():
                    try:
                        pdfmetrics.registerFont(TTFont("ResearchCN", str(path)))
                        break
                    except Exception:
                        continue
            else:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                return "STSong-Light"
        return "ResearchCN"


def _inline(text):
    text = escape(text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-"))
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<link href="\2" color="#146A83">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r'<font color="#163D57">\1</font>', text)
    return text.replace("`", "")


def _story(markdown, font, body_size):
    body = ParagraphStyle("body", fontName=font, fontSize=body_size, leading=body_size * 1.45,
                          textColor=colors.HexColor("#243747"), spaceAfter=4, wordWrap="CJK", splitLongWords=True)
    styles = {level: ParagraphStyle(f"h{level}", parent=body, fontSize=size,
                                    leading=size * 1.35, spaceBefore=10 if level > 1 else 0,
                                    spaceAfter=6, keepWithNext=True, textColor=colors.HexColor("#123A54"))
              for level, size in [(1, 20), (2, 14), (3, 11.3), (4, 10.5)]}
    table_style = ParagraphStyle("table", parent=body, fontSize=9.2, leading=12.5, spaceAfter=0)
    label_style = ParagraphStyle("label", parent=body, keepWithNext=True)
    subtitle_style = ParagraphStyle("subtitle", parent=body, fontSize=12.0, leading=17,
                                    spaceAfter=8, keepWithNext=True, textColor=colors.HexColor("#48616D"))
    story, lines, i = [], markdown.splitlines(), 0
    after_title = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        is_subtitle = after_title and not line.startswith(("#", ">", "|", "**", "- "))
        after_title = False
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))
                if not all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells):
                    rows.append([Paragraph(_inline(cell.strip().replace("\\|", "|")), table_style) for cell in cells])
                i += 1
            if rows:
                cols = max(len(row) for row in rows)
                rows = [row + [""] * (cols - len(row)) for row in rows]
                table = Table(rows, colWidths=[(A4[0] - 92) / cols] * cols, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F0F3")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#A9C1CB")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story += [table, Spacer(1, 6)]
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading:
            story.append(Paragraph(_inline(heading[2]), styles[len(heading[1])]))
            after_title = heading[1] == "#" and heading[2].endswith(" 投资报告")
        else:
            line = re.sub(r"^>\s?", "", line)
            style = label_style if re.fullmatch(r"\*\*[^*]+\*\*", line) else body
            story.append(Paragraph(_inline(line), subtitle_style if is_subtitle else style))
        i += 1
    return story


def build_report_pdf(markdown: str, max_pages: int = DEFAULT_REPORT_MAX_PAGES) -> tuple[bytes, int]:
    """No truncation or unreadable font shrinkage; oversize reports are rejected."""
    if not 1 <= max_pages <= REPORT_PAGE_CAP:
        raise ValueError(f"报告页数上限须在 1 到 {REPORT_PAGE_CAP} 之间")
    font = _font()
    heading = re.search(r"(?m)^#\s+(.+)$", markdown)
    title = heading.group(1).strip() if heading else "投资报告"
    last_count = 0
    for body_size in (10.5, 10.0):
        stream = BytesIO()
        doc = SimpleDocTemplate(stream, pagesize=A4, rightMargin=46, leftMargin=46,
                                topMargin=46, bottomMargin=44, title=title, author="投资研究团队")

        def decorate(canvas, document):
            canvas.saveState()
            canvas.setFont(font, 8.3)
            canvas.setFillColor(colors.HexColor("#667E8C"))
            canvas.drawString(46, A4[1] - 27, "投资研究  /  内部讨论")
            canvas.drawRightString(A4[0] - 46, 25, f"{document.page}")
            canvas.restoreState()

        try:
            doc.build(_story(markdown, font, body_size), onFirstPage=decorate, onLaterPages=decorate)
        except LayoutError as exc:
            raise ReportPageLimitError("报告中的表格或段落无法在可读版面内排下，需压缩正文后重新核查") from exc
        payload = stream.getvalue()
        last_count = len(PdfReader(BytesIO(payload)).pages)
        if 1 <= last_count <= max_pages:
            return payload, last_count
    raise ReportPageLimitError(f"报告实际为 {last_count} 页，超过 {max_pages} 页上限；需压缩正文后重新核查")
