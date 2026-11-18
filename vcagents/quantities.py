"""数值标准化：把“148 亿”“USD 14.8 billion”“前四个月”等写法换算成同一数值再比较。

主张一侧只取阿拉伯数字（可带量级）；原文一侧另外识别中文数字、英文数词和英文月份。
这样主张写法换算后仍能核对，而原文中的普通汉字、英文单词不会让主张多出必须核对的数字。
只做确定的写法换算，不推断币种、汇率、口径或语义。
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

_WIDTH = str.maketrans("０１２３４５６７８９．％，＋－−﹣", "0123456789.%,+---")
_REFERENCE = re.compile(r"\[(?:E-(?:[A-Za-z0-9]+-)?\d+|(?:C\d+|S)-\d+|U-\d+)\]")
_SCALES = {
    "万亿": 1e12, "千亿": 1e11, "百亿": 1e10, "十亿": 1e9, "亿": 1e8, "千万": 1e7, "百万": 1e6, "十万": 1e5, "万": 1e4,
    "trillion": 1e12, "billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6, "thousand": 1e3,
    "t": 1e12, "b": 1e9, "m": 1e6, "k": 1e3,
}
_CN_SCALE = "万亿|千亿|百亿|十亿|亿|千万|百万|十万|万"
_EN_SCALE = "trillion|billion|bn|million|mn|thousand"
# 数字前不能紧跟字母或数字（排除 B2B、Veo3 这类名称）；单字母量级必须紧贴数字，避免“2026 B轮”被读成 2026 × 10^9
_ARABIC = re.compile(
    r"(?<![A-Za-z\d.])([+-]?\d+(?:,\d{3}(?!\d))*(?:\.\d+)?)"
    rf"(?:\s*({_CN_SCALE})|\s*({_EN_SCALE})\b|([TBMKk])(?![A-Za-z]))?",
    re.IGNORECASE,
)
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_RUN = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")
_EN_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12,
}
_EN_NUMBER = re.compile(
    rf"\b({'|'.join(_EN_WORDS)})(?:[-\s]({'|'.join(k for k, v in _EN_WORDS.items() if v < 10)}))?\b(?:\s+({_EN_SCALE})\b)?",
    re.IGNORECASE,
)
_MONTHS = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8,
           "September": 9, "October": 10, "November": 11, "December": 12, "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
           "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12}
_MONTH = re.compile(rf"\b({'|'.join(sorted(_MONTHS, key=len, reverse=True))})\b")
# 真实反例：原文 “tenfold since June”，主张写“增长 10 倍”
_FOLD = re.compile(rf"\b({'|'.join(_EN_WORDS)})[-\s]?fold\b", re.IGNORECASE)
_MULTIPLE_WORDS = {"double": 2, "doubled": 2, "doubles": 2, "doubling": 2, "triple": 3, "tripled": 3, "tripling": 3,
                   "quadrupled": 4, "quadrupling": 4, "quintupled": 5}
_MULTIPLE = re.compile(rf"\b({'|'.join(_MULTIPLE_WORDS)})\b", re.IGNORECASE)


def loose_text(text: str) -> str:
    """仅作名称/文字的宽松比较；金额与比例须解析数值，不能用此函数忽略小数点和正负号。"""
    return re.sub(r"[^\w%]|_", "", unicodedata.normalize("NFKC", str(text)).casefold())


@dataclass(frozen=True)
class Quantity:
    text: str  # 原文写法，例如“148 亿”
    value: float  # 换算后的数值
    start: int = 0
    end: int = 0
    scaled: bool = False  # 带量级（亿、万、million 等）


def claim_quantities(text: str) -> list[Quantity]:
    """主张中的阿拉伯数字及其量级。"""
    text = _REFERENCE.sub(lambda m: " " * len(m.group()), text.translate(_WIDTH))
    out = []
    for match in _ARABIC.finditer(text):
        scale = match.group(2) or match.group(3) or match.group(4)
        factor = _SCALES.get((scale or "").lower(), 1)
        out.append(Quantity(match.group(0).strip(), float(match.group(1).replace(",", "")) * factor,
                            match.start(), match.end(), bool(scale)))
    return out


_UNIT_AFTER = re.compile(r"\s*(?:\+\s*)?(%|倍|x(?![A-Za-z])|×|美元|元|人民币|港元|欧元|USD|RMB|CNY|用户|名用户|家客户|"
                         r"(?:个|种)(?:(?:以上的|以上|顶级|第三方|AI|视频|图像|生成式)\s*)*模型)", re.IGNORECASE)
_UNIT_BEFORE = re.compile(r"(\$|US\$|USD|￥|¥|RMB|CNY|€)\s*$", re.IGNORECASE)
_MONEY_UNIT = (r"(?:万亿|亿|千万|百万|万)\s*(?:美元|人民币|港元|港币|欧元|英镑|日元)"
               r"|(?:USD|CNY|RMB|HKD|EUR|GBP|JPY)\s+(?:thousand|million|billion)s?\b")
_TABLE_UNIT = re.compile(rf"(?:金额单位|单位)\s*[：:]\s*({_MONEY_UNIT})|(?<![\w])({_MONEY_UNIT})", re.I)


def table_unit_label(line: str) -> re.Match | None:
    """Recognize an explicit table unit without treating a prose amount as a header."""
    label = _TABLE_UNIT.search(line)
    if label and (label[1] or (re.search(r"[|\t（(]|\bin\s+", line, re.I)
                               and not any(q.scaled for q in claim_quantities(line)))):
        return label
    return None


def fact_quantities(text: str) -> list[Quantity]:
    """事实性数量：带量级、币种、百分号或倍数的数字；不含年份、序号、评分等普通数字。"""
    normalized = text.translate(_WIDTH)
    return [q for q in claim_quantities(text)
            if q.scaled or _UNIT_AFTER.match(normalized, q.end) or _UNIT_BEFORE.search(normalized[:q.start])]


def same_value(a: float, b: float) -> bool:
    return _same(a, b)


def display_matches(text: str, exact: float) -> bool:
    """Compare a displayed quantity with the exact value at the precision of its written unit."""
    quantity = claim_quantities(text)[0]
    number = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", quantity.text)
    digits = number.group().replace(",", "")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    scale = claim_quantities("1" + quantity.text[number.end():])[0].value
    if "%" in text or "％" in text:
        exact *= 100
    return abs(quantity.value - exact) <= 0.5 * 10 ** -decimals * scale + 1e-12


def source_values(text: str) -> set[float]:
    """原文中的规范数值；带量级时只保留换算值，避免把1美元匹配到1 million。"""
    text = text.translate(_WIDTH)
    # A header saying 百万美元 specifies how to read cells, and contains no amount by itself.
    lines = []
    for line in text.splitlines():
        if label := table_unit_label(line):
            start, end = label.span(label.lastindex)
            line = line[:start] + " " + line[end:]
        lines.append(line)
    text = "\n".join(lines)
    values: set[float] = set()
    used_chinese_scales = set()
    for match in _ARABIC.finditer(text):
        raw = float(match.group(1).replace(",", ""))
        scale = match.group(2) or match.group(3) or match.group(4)
        values.add(raw * _SCALES.get((scale or "").lower(), 1))
        if match.group(2):
            used_chinese_scales.add(match.start(2))
    for match in _CN_RUN.finditer(text):
        # For 100 万, the Arabic parser already applied 万 to 100.
        if match.start() in used_chinese_scales:
            continue
        value = _chinese_number(match.group(0))
        if value is not None:
            values.add(value)
    for match in _EN_NUMBER.finditer(text):
        first = _EN_WORDS[match.group(1).lower()]
        second = _EN_WORDS[match.group(2).lower()] if match.group(2) else 0
        if 20 <= first < 100:  # twenty-five
            parts = [first + second]
        else:  # “one two” 不是一个数
            parts = [first] + ([second] if second else [])
        if match.group(3):
            values.add(parts[-1] * _SCALES[match.group(3).lower()])
        else:
            values.update(float(p) for p in parts)
    values.update(float(_MONTHS[m.group(1)]) for m in _MONTH.finditer(text))
    values.update(float(_EN_WORDS[m.group(1).lower()]) for m in _FOLD.finditer(text))
    values.update(float(_MULTIPLE_WORDS[m.group(1).lower()]) for m in _MULTIPLE.finditer(text))
    return values


def unmatched(claim: str, *sources: str) -> list[str]:
    """主张中在原文里找不到等值写法的数量，按主张原写法返回。"""
    values = set().union(*(source_values(s) for s in sources)) if sources else set()
    return [q.text for q in claim_quantities(claim) if not any(_same(q.value, v) for v in values)]


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _chinese_number(run: str) -> float | None:
    """“三千二百万”“千万”“四”→ 数值；无法解析的组合返回 None。"""
    total, section, number = 0.0, 0.0, 0.0
    for ch in run:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            section += (number or 1) * _CN_UNITS[ch]
            number = 0
        elif ch == "万":
            total += ((section + number) or 1) * 1e4
            section = number = 0
        elif ch == "亿":
            total = ((total + section + number) or 1) * 1e8
            section = number = 0
        else:
            return None
    return total + section + number
