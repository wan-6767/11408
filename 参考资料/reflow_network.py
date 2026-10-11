#!/usr/bin/env python3
"""Build question/answer-only study PDFs from the scanned Wangdao textbooks.

The source page is rendered to RGB pixels and every block is cropped from that
render.  The text layer is used only to locate block boundaries; text, figures,
formulas, and answer diagrams are never re-typeset.
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SOURCE_WIDTH_PT = 541.0
SOURCE_HEIGHT_PT = 754.0
SOURCE_X_MIN_PT = 52.0
SOURCE_X_MAX_PT = 490.0
CONTENT_TOP_PT = 64.0
CONTENT_BOTTOM_PT = 706.0
BOUNDARY_GAP_PT = 0.5
END_BOUNDARY_GAP_PT = 4.0
QUESTION_ANSWER_GAP_PT = 14.0
ANSWER_QUESTION_GAP_PT = 8.0
CHAPTER_SECTION_GAP_PT = 3.0
SECTION_QUESTION_GAP_PT = 8.0
PAGE_MARGIN_PT = 14.0
CROP_PADDING_PT = 4.0
BLOCK_BOTTOM_PADDING_PT = 1.25
MAX_BOUNDARY_EXTENSION_PT = 6.0
DEFAULT_DPI = 250
RENDER_CACHE_SIZE = 8

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
HEADING_RE = re.compile(
    r"^\s*[＊*]?\s*([０-９0-9]+)[．.]([０-９0-9]+)[．.]([０-９0-9]+)\s*"
    r"(本节(?:习题|试题)精选|答案与解析)"
)
NUMBER_RE = re.compile(r"^\s*([０-９0-9]{2})[．.]")
KNOWLEDGE_HEADING_RE = re.compile(
    r"^\s*[＊*]?\s*[１-９1-9][．.][１-９1-9](?:[．.][１-９1-9])?"
    r"(?![０-９0-9．.])"
)
SECTION_TITLE_RE = re.compile(
    r"^\s*[＊*]?\s*([０-９0-9]+)[．.]([０-９0-9]+)(?![０-９0-9．.])"
)
CHAPTER_TITLE_RE = re.compile(r"^\s*第([０-９0-9]+)章")


@dataclass(frozen=True)
class TextLine:
    y: float
    y_max: float
    x: float
    x_max: float
    text: str


@dataclass(frozen=True)
class Heading:
    prefix: str
    kind: str
    page: int
    y: float
    text: str
    y_max: float = 0.0
    x: float = 0.0
    x_max: float = 0.0


@dataclass(frozen=True)
class Start:
    page: int
    y: float
    ordinal: int
    source_number: int
    y_max: float = 0.0
    x: float = 0.0
    x_max: float = 0.0
    text: str = ""


@dataclass(frozen=True)
class Boundary:
    page: int
    y: float
    text: str = ""
    y_max: float = 0.0
    x: float = 0.0
    x_max: float = 0.0


@dataclass(frozen=True)
class Segment:
    page: int
    top: float
    bottom: float
    special: Optional[str] = None


@dataclass(frozen=True)
class ExerciseSection:
    prefix: str
    section_title: Boundary
    chapter_title: Boundary
    question_starts: tuple[Start, ...]
    answer_starts: tuple[Start, ...]
    question_end: Boundary
    answer_end: Boundary


@dataclass(frozen=True)
class BookConfig:
    key: str
    label: str
    content_start_page: int
    expected_sections: int
    special_regions: tuple[tuple[str, str, int], ...] = ()


BOOK_CONFIGS = {
    "network": BookConfig(
        key="network",
        label="计算机网络",
        content_start_page=12,
        expected_sections=28,
        special_regions=(("１.２", "q36", 35), ("１.２", "q37", 36)),
    ),
    "os": BookConfig(
        key="os",
        label="操作系统",
        content_start_page=12,
        expected_sections=16,
        special_regions=(
            ("３.１", "os_q7", 6),
            ("５.２", "os_q38", 37),
        ),
    ),
    "ds": BookConfig(
        key="ds",
        label="数据结构",
        content_start_page=12,
        expected_sections=30,
        special_regions=(
            ("５.３", "ds_q32", 31),
            ("５.３", "ds_q36", 35),
            ("５.３", "ds_q37", 36),
            ("５.３", "ds_a05", 4),
            ("５.３", "ds_a42", 41),
            ("５.４", "ds_q12", 11),
            ("５.４", "ds_a20", 19),
            ("５.５", "ds_a18", 17),
            ("５.５", "ds_a23", 22),
            ("６.２", "ds_q05", 4),
            ("６.２", "ds_a21", 20),
            ("６.２", "ds_a22", 21),
            ("６.３", "ds_q07", 6),
            ("６.４", "ds_q15", 14),
            ("６.４", "ds_q16", 15),
            ("６.４", "ds_q26", 25),
            ("６.４", "ds_q49", 48),
            ("６.４", "ds_q58", 57),
            ("７.２", "ds_q21", 20),
        ),
    ),
    "co": BookConfig(
        key="co",
        label="计算机组成原理",
        content_start_page=12,
        expected_sections=27,
        special_regions=(
            ("２.３", "co_q49", 48),
            ("５.３", "co_a20", 19),
        ),
    ),
}


class PPMPage:
    def __init__(self, path: Path):
        self.width, self.height, self.data = self._read(path)

    @staticmethod
    def _read(path: Path) -> tuple[int, int, bytes]:
        raw = path.read_bytes()
        pos = 0

        def token() -> bytes:
            nonlocal pos
            while pos < len(raw):
                if raw[pos] in b" \t\r\n":
                    pos += 1
                    continue
                if raw[pos] == 35:
                    while pos < len(raw) and raw[pos] not in b"\r\n":
                        pos += 1
                    continue
                break
            start = pos
            while pos < len(raw) and raw[pos] not in b" \t\r\n":
                pos += 1
            return raw[start:pos]

        magic = token()
        width = int(token())
        height = int(token())
        maxval = int(token())
        if magic != b"P6" or maxval != 255:
            raise ValueError(f"unsupported PPM format: {path}")
        while pos < len(raw) and raw[pos] in b" \t\r\n":
            pos += 1
        data = raw[pos:]
        expected = width * height * 3
        if len(data) != expected:
            raise ValueError(f"bad PPM payload: {path} ({len(data)} != {expected})")
        return width, height, data


class RenderCache:
    """Render source pages lazily and retain only a small LRU cache."""

    def __init__(self, source: Path, dpi: int, temp_dir: Path):
        self.source = source
        self.dpi = dpi
        self.temp_dir = temp_dir
        self.pages: OrderedDict[int, tuple[Path, PPMPage]] = OrderedDict()

    def get(self, page: int) -> PPMPage:
        cached = self.pages.pop(page, None)
        if cached is not None:
            self.pages[page] = cached
            return cached[1]

        ppm_path = self.temp_dir / f"source-{page:03d}.ppm"
        prefix = self.temp_dir / f"render-{page:03d}"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(self.dpi),
                "-singlefile",
                str(self.source),
                str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        rendered = prefix.with_suffix(".ppm")
        rendered.replace(ppm_path)
        item = (ppm_path, PPMPage(ppm_path))
        self.pages[page] = item
        while len(self.pages) > RENDER_CACHE_SIZE:
            _, (old_path, _) = self.pages.popitem(last=False)
            old_path.unlink(missing_ok=True)
        return item[1]


class TextIndex:
    def __init__(self, pages: list[list[TextLine]]):
        self.pages = pages

    def page_lines(self, page: int) -> list[TextLine]:
        return self.pages[page - 1]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def build_text_index(source: Path, temp_dir: Path) -> TextIndex:
    bbox = temp_dir / "bbox.xml"
    subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-bbox-layout", str(source), str(bbox)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    root = ET.parse(bbox).getroot()
    pages: list[list[TextLine]] = []
    for page in (e for e in root.iter() if local_name(e.tag) == "page"):
        lines: list[TextLine] = []
        for line in (e for e in page.iter() if local_name(e.tag) == "line"):
            words = [w.text or "" for w in line if local_name(w.tag) == "word"]
            lines.append(
                TextLine(
                    y=float(line.attrib["yMin"]),
                    y_max=float(line.attrib["yMax"]),
                    x=float(line.attrib["xMin"]),
                    x_max=float(line.attrib["xMax"]),
                    text="".join(words),
                )
            )
        lines.sort(key=lambda item: (item.y, item.x))
        pages.append(lines)
    return TextIndex(pages)


def find_headings(index: TextIndex, config: BookConfig) -> list[Heading]:
    result: list[Heading] = []
    for page, lines in enumerate(index.pages, 1):
        # Early pages contain the table of contents and repeat all exercise
        # headings. The actual content starts after the front matter.  A
        # source heading in the contents also contains dot leaders, whereas a
        # real section heading does not.
        if page < config.content_start_page:
            continue
        for line in lines:
            match = HEADING_RE.match(line.text)
            if match and "…" not in line.text:
                result.append(
                    Heading(
                        prefix=".".join(match.group(i) for i in (1, 2)),
                        kind=match.group(4),
                        page=page,
                        y=line.y,
                        text=line.text,
                        y_max=line.y_max,
                        x=line.x,
                        x_max=line.x_max,
                    )
                )
    return result


def _candidate_number(line: TextLine) -> Optional[tuple[int, int, str]]:
    if line.x > 110.0:
        return None
    match = NUMBER_RE.match(line.text)
    if not match:
        return None
    number = int(match.group(1).translate(FULLWIDTH_DIGITS))
    return number, match.end(), line.text[match.end() :]


def collect_starts(
    index: TextIndex,
    first: Boundary,
    last: Boundary,
    *,
    answers: bool,
) -> list[Start]:
    candidates: list[tuple[int, TextLine, int, str]] = []
    for page in range(first.page, last.page + 1):
        for line in index.page_lines(page):
            if not ((page > first.page or line.y > first.y) and (page < last.page or line.y < last.y)):
                continue
            item = _candidate_number(line)
            if item is not None:
                number, end, tail = item
                candidates.append((page, line, number, tail))

    # Exercise lists are numbered 01..N and, when present, a second list of
    # comprehensive questions restarts at 01.  Numeric text in diagrams and
    # explanations can also look like "35.", so only the next expected number
    # is accepted.  Comprehensive answer blocks are explicitly marked by
    # "【解答】", which makes their reset unambiguous.
    result: list[Start] = []
    expected = 1
    for page, line, number, tail in candidates:
        if number == expected:
            result.append(
                Start(
                    page,
                    line.y,
                    len(result),
                    number,
                    line.y_max,
                    line.x,
                    line.x_max,
                    line.text,
                )
            )
            expected += 1
            continue
        tail_start = tail.lstrip()
        if number == 1 and expected > 1:
            if (answers and tail_start.startswith("【")) or (
                not answers and tail_start and tail_start[0] not in "０１２３４５６７８９0123456789"
            ):
                result.append(
                    Start(
                        page,
                        line.y,
                        len(result),
                        number,
                        line.y_max,
                        line.x,
                        line.x_max,
                        line.text,
                    )
                )
                expected = 2
            continue
        # A few source sections have omitted comprehensive-question numbers
        # (for example 01..06 followed by 09).  The bracketed exam marker is
        # a reliable boundary, so retain the printed number instead of
        # inventing or renumbering the question.
        if not answers and tail_start.startswith("【") and number > expected:
            result.append(
                Start(
                    page,
                    line.y,
                    len(result),
                    number,
                    line.y_max,
                    line.x,
                    line.x_max,
                    line.text,
                )
            )
            expected = number + 1
    return result


def first_summary_after(index: TextIndex, boundary: Boundary) -> Optional[Boundary]:
    summary_markers = (
        "本章小结及疑难点",
        "本章小结",
        "本章总结",
        "归纳总结",
        "参考文献",
    )
    for page in range(boundary.page, len(index.pages) + 1):
        for line in index.page_lines(page):
            if page == boundary.page and line.y <= boundary.y:
                continue
            if any(marker in line.text for marker in summary_markers):
                return Boundary(page, line.y, line.text, line.y_max, line.x, line.x_max)
    return None


def first_knowledge_heading_after(index: TextIndex, boundary: Boundary) -> Optional[Boundary]:
    """Find the next prose-section heading after an answer block.

    The next exercise may be several pages later.  Ending at that exercise
    heading would pull the intervening knowledge-point pages into the answer.
    """
    for page in range(boundary.page, len(index.pages) + 1):
        for line in index.page_lines(page):
            if page == boundary.page and line.y <= boundary.y:
                continue
            if line.x <= 120.0 and KNOWLEDGE_HEADING_RE.match(line.text):
                return Boundary(page, line.y, line.text, line.y_max, line.x, line.x_max)
    return None


def prefix_key(prefix: str) -> str:
    """Normalize section prefixes for comparisons across fullwidth variants."""
    return prefix.translate(FULLWIDTH_DIGITS).replace("．", ".")


def section_prefix_key(text: str) -> Optional[str]:
    match = SECTION_TITLE_RE.match(text)
    if match is None:
        return None
    return ".".join(group.translate(FULLWIDTH_DIGITS) for group in match.groups())


KNOWLEDGE_TITLE_RE = re.compile(
    r"^\s*[＊*]?\s*([０-９0-9]+)[．.]([０-９0-9]+)[．.]([０-９0-9]+)"
)


def knowledge_prefix_key(text: str) -> Optional[str]:
    match = KNOWLEDGE_TITLE_RE.match(text)
    if match is None:
        return None
    return ".".join(group.translate(FULLWIDTH_DIGITS) for group in match.groups())


def fallback_section_title(index: TextIndex, heading: Heading) -> Optional[Boundary]:
    """Find a source title row whose numeric badge is partly graphical.

    Some chapter opener pages render the section badge as vector artwork. The
    text layer then exposes only the title (or a fragment such as ``．４``),
    so an exact ``3.4`` match is unavailable. The first numbered subsection
    gives a reliable anchor: the section title is the short row immediately
    before it on the same source page.
    """
    target = prefix_key(heading.prefix)
    subsection: Optional[Boundary] = None
    for page in range(heading.page, 0, -1):
        lines = index.page_lines(page)
        for line in lines:
            if page == heading.page and line.y >= heading.y:
                continue
            key = knowledge_prefix_key(line.text)
            if key == f"{target}.1":
                subsection = Boundary(page, line.y, line.text, line.y_max, line.x, line.x_max)
                break
        if subsection is not None:
            break
    if subsection is None:
        return None

    lines = index.page_lines(subsection.page)
    candidates = [
        line
        for line in lines
        if line.y < subsection.y
        and subsection.y - line.y <= 120.0
        and line.x <= 120.0
        and line.text.strip()
        and len(line.text.strip()) <= 40
        and not any(mark in line.text for mark in ("，", "。", "；", "："))
        and not any(marker in line.text for marker in ("考点", "考纲", "知识框架", "复习提示"))
        and not line.text.lstrip().startswith(("第", "图", "表"))
    ]
    if not candidates:
        return None
    line = max(candidates, key=lambda item: item.y)
    # Use the full content width for a graphical badge/underline so the
    # locator keeps pixels that are absent from the text layer.
    return Boundary(
        subsection.page,
        line.y,
        line.text,
        line.y_max,
        SOURCE_X_MIN_PT,
        SOURCE_X_MAX_PT,
    )


def find_section_title(index: TextIndex, heading: Heading) -> Optional[Boundary]:
    """Find the nearest source section title preceding a question heading."""
    target = prefix_key(heading.prefix)
    for page in range(heading.page, 0, -1):
        lines = index.page_lines(page)
        for line in reversed(lines):
            if page == heading.page and line.y >= heading.y:
                continue
            if line.x > 120.0:
                continue
            if section_prefix_key(line.text) == target and "…" not in line.text:
                return Boundary(page, line.y, line.text, line.y_max, line.x, line.x_max)
    return fallback_section_title(index, heading)


def find_chapter_title(
    index: TextIndex,
    page_start: int,
    prefix: str,
) -> Optional[Boundary]:
    """Find a compact running chapter title near the first question page."""
    target = prefix_key(prefix).split(".", 1)[0]
    page_count = len(index.pages)
    offsets = [0, 1, -1, 2, -2, 3, -3]
    pages = [
        page_start + offset
        for offset in offsets
        if 1 <= page_start + offset <= page_count
    ]
    for page in pages:
        lines = index.page_lines(page)
        for line in reversed(lines):
            if line.y >= CONTENT_TOP_PT:
                continue
            match = CHAPTER_TITLE_RE.match(line.text)
            if match is None:
                continue
            chapter = match.group(1).translate(FULLWIDTH_DIGITS)
            if chapter == target:
                return Boundary(page, line.y, line.text, line.y_max, line.x, line.x_max)
    return None


def build_sections(index: TextIndex, config: BookConfig) -> list[ExerciseSection]:
    headings = find_headings(index, config)
    sections: list[ExerciseSection] = []
    for position, heading in enumerate(headings):
        if heading.kind not in ("本节习题精选", "本节试题精选"):
            continue
        answer_heading = next(
            (
                candidate
                for candidate in headings[position + 1 :]
                if candidate.prefix == heading.prefix and candidate.kind == "答案与解析"
            ),
            None,
        )
        if answer_heading is None:
            raise ValueError(f"missing answer heading for {heading.prefix}")

        next_question_heading = next(
            (
                candidate
                for candidate in headings[position + 1 :]
                if candidate.kind in ("本节习题精选", "本节试题精选")
            ),
            None,
        )
        answer_boundary = Boundary(
            answer_heading.page,
            answer_heading.y,
            answer_heading.text,
            answer_heading.y_max,
            answer_heading.x,
            answer_heading.x_max,
        )
        possible_ends = []
        if next_question_heading is not None:
            possible_ends.append(
                Boundary(
                    next_question_heading.page,
                    next_question_heading.y,
                    next_question_heading.text,
                    next_question_heading.y_max,
                    next_question_heading.x,
                    next_question_heading.x_max,
                )
            )
            # The next subsection title belongs to the locator strip of the
            # next pair. Exclude that source row from the current answer so
            # it is not duplicated at a chapter or subsection boundary.
            next_section_title = find_section_title(index, next_question_heading)
            if next_section_title is not None and (
                next_section_title.page > answer_boundary.page
                or (
                    next_section_title.page == answer_boundary.page
                    and next_section_title.y > answer_boundary.y
                )
            ):
                possible_ends.append(next_section_title)
        knowledge_heading = first_knowledge_heading_after(index, answer_boundary)
        if knowledge_heading is not None:
            possible_ends.append(knowledge_heading)
        summary = first_summary_after(index, answer_boundary)
        if summary is not None:
            possible_ends.append(summary)
        answer_end = min(possible_ends, key=lambda item: (item.page, item.y)) if possible_ends else Boundary(len(index.pages), 706.0, "EOF")

        question_starts = collect_starts(
            index,
            Boundary(heading.page, heading.y, heading.text),
            answer_boundary,
            answers=False,
        )
        answer_starts = collect_starts(index, answer_boundary, answer_end, answers=True)
        if len(question_starts) != len(answer_starts):
            raise ValueError(
                f"{heading.prefix}: question/answer count mismatch "
                f"{len(question_starts)} != {len(answer_starts)}"
            )
        if not question_starts:
            raise ValueError(f"{heading.prefix}: no questions found")
        section_title = find_section_title(index, heading)
        if section_title is None:
            raise ValueError(f"{heading.prefix}: missing source section title")
        chapter_title = find_chapter_title(index, question_starts[0].page, heading.prefix)
        if chapter_title is None:
            raise ValueError(f"{heading.prefix}: missing source chapter title")
        sections.append(
            ExerciseSection(
                prefix=heading.prefix,
                section_title=section_title,
                chapter_title=chapter_title,
                question_starts=tuple(question_starts),
                answer_starts=tuple(answer_starts),
                question_end=answer_boundary,
                answer_end=answer_end,
            )
        )
    if len(sections) != config.expected_sections:
        raise ValueError(
            f"{config.label}: expected {config.expected_sections} exercise sections, "
            f"found {len(sections)}"
        )
    return sections


def point_to_px(page: PPMPage, x: float, y: float) -> tuple[int, int]:
    return round(x * page.width / SOURCE_WIDTH_PT), round(y * page.height / SOURCE_HEIGHT_PT)


def crop(page: PPMPage, top: float, bottom: float) -> tuple[int, int, bytes]:
    x0, _ = point_to_px(page, SOURCE_X_MIN_PT, 0)
    x1, _ = point_to_px(page, SOURCE_X_MAX_PT, 0)
    _, y0 = point_to_px(page, 0, top)
    _, y1 = point_to_px(page, 0, bottom)
    x0 = max(0, min(page.width, x0))
    x1 = max(x0, min(page.width, x1))
    y0 = max(0, min(page.height, y0))
    y1 = max(y0, min(page.height, y1))
    width = x1 - x0
    row_size = page.width * 3
    return width, y1 - y0, b"".join(
        page.data[row * row_size + x0 * 3 : row * row_size + x1 * 3]
        for row in range(y0, y1)
    )


def crop_title_line(page: PPMPage, boundary: Boundary) -> tuple[int, int, bytes]:
    """Place one source title line in a full-width white strip."""
    base_x0, _ = point_to_px(page, SOURCE_X_MIN_PT, 0)
    base_x1, _ = point_to_px(page, SOURCE_X_MAX_PT, 0)
    x0, _ = point_to_px(page, max(SOURCE_X_MIN_PT, boundary.x - CROP_PADDING_PT), 0)
    x1, _ = point_to_px(page, min(SOURCE_X_MAX_PT, boundary.x_max + CROP_PADDING_PT), 0)
    _, y0 = point_to_px(page, 0, max(0.0, boundary.y - CROP_PADDING_PT))
    _, y1 = point_to_px(page, 0, min(SOURCE_HEIGHT_PT, boundary.y_max + CROP_PADDING_PT))
    width = base_x1 - base_x0
    height = max(1, y1 - y0)
    row_size = page.width * 3
    out = bytearray(b"\xff" * (width * height * 3))
    if x1 <= x0 or y1 <= y0:
        return width, height, bytes(out)
    copy_width = x1 - x0
    for source_y in range(y0, y1):
        dst_y = source_y - y0
        src_off = source_y * row_size + x0 * 3
        dst_off = dst_y * width * 3 + (x0 - base_x0) * 3
        out[dst_off : dst_off + copy_width * 3] = page.data[
            src_off : src_off + copy_width * 3
        ]
    return width, height, bytes(out)


def crop_q36(page: PPMPage) -> tuple[int, int, bytes]:
    """Keep the left question text and right timing diagram separately."""
    # The previous question's option row ends at about y=346; start just
    # before q36's number so no fragments from q35 enter the crop.
    top, left_bottom, bottom = 348.0, 409.0, 455.0
    base_x0, _ = point_to_px(page, SOURCE_X_MIN_PT, 0)
    base_x1, _ = point_to_px(page, SOURCE_X_MAX_PT, 0)
    _, y0 = point_to_px(page, 0, top)
    _, y_left = point_to_px(page, 0, left_bottom)
    _, y1 = point_to_px(page, 0, bottom)
    x_left0, _ = point_to_px(page, 70.0, 0)
    x_left1, _ = point_to_px(page, 360.0, 0)
    x_right0, _ = point_to_px(page, 350.0, 0)
    x_right1, _ = point_to_px(page, 500.0, 0)
    x_left0 = max(base_x0, x_left0)
    x_left1 = min(base_x1, x_left1)
    x_right0 = max(base_x0, x_right0)
    x_right1 = min(base_x1, x_right1)
    width = base_x1 - base_x0
    height = y1 - y0
    row_size = page.width * 3
    out = bytearray(b"\xff" * (width * height * 3))

    def copy_rect(src_y0: int, src_y1: int, x0: int, x1: int) -> None:
        rect_width = x1 - x0
        for src_y in range(src_y0, src_y1):
            dst_y = src_y - y0
            src_off = src_y * row_size + x0 * 3
            dst_off = dst_y * width * 3 + (x0 - base_x0) * 3
            out[dst_off : dst_off + rect_width * 3] = page.data[src_off : src_off + rect_width * 3]

    copy_rect(y0, y_left, x_left0, x_left1)
    copy_rect(y0, y1, x_right0, x_right1)
    return width, height, bytes(out)


def crop_q37(page: PPMPage) -> tuple[int, int, bytes]:
    """Remove the part of q36's timing diagram below q37's start line."""
    top, bottom = 410.0, 469.45
    width, height, data = crop(page, top, bottom)
    _, mask_y0 = point_to_px(page, 0, 410.0)
    _, mask_y1 = point_to_px(page, 0, 455.0)
    base_x0, _ = point_to_px(page, SOURCE_X_MIN_PT, 0)
    x0, _ = point_to_px(page, 360.0, 0)
    x0 = max(0, x0 - base_x0)
    row_size = width * 3
    out = bytearray(data)
    for source_y in range(mask_y0, mask_y1):
        row = source_y - mask_y0
        if 0 <= row < height:
            start = row * row_size + x0 * 3
            out[start : (row + 1) * row_size] = b"\xff" * (row_size - x0 * 3)
    return width, height, bytes(out)


def crop_masked(
    page: PPMPage,
    top: float,
    bottom: float,
    masks: tuple[tuple[float, float, float, float], ...] = (),
) -> tuple[int, int, bytes]:
    """Crop a source region and white-mask known neighboring-column content."""
    width, height, data = crop(page, top, bottom)
    row_size = width * 3
    out = bytearray(data)
    base_x0, _ = point_to_px(page, SOURCE_X_MIN_PT, 0)
    for mask_top, mask_bottom, mask_x0_pt, mask_x1_pt in masks:
        _, mask_y0 = point_to_px(page, 0, mask_top)
        _, mask_y1 = point_to_px(page, 0, mask_bottom)
        mask_y0 = max(0, mask_y0 - point_to_px(page, 0, top)[1])
        mask_y1 = min(height, mask_y1 - point_to_px(page, 0, top)[1])
        mask_x0, _ = point_to_px(page, mask_x0_pt, 0)
        mask_x1, _ = point_to_px(page, mask_x1_pt, 0)
        mask_x0 = max(0, mask_x0 - base_x0)
        mask_x1 = min(width, mask_x1 - base_x0)
        if mask_y1 <= mask_y0 or mask_x1 <= mask_x0:
            continue
        blank = b"\xff" * ((mask_x1 - mask_x0) * 3)
        for row in range(mask_y0, mask_y1):
            start = row * row_size + mask_x0 * 3
            out[start : start + len(blank)] = blank
    return width, height, bytes(out)


def crop_special(page: PPMPage, special: str) -> tuple[int, int, bytes]:
    """Apply a source-specific crop for a cross-column or early graphic."""
    rules: dict[str, tuple[float, float, tuple[tuple[float, float, float, float], ...]]] = {
        # The memory map in q07 continues below q08's left-column text.
        "os_q7": (285.0, 386.0, ((346.45, 386.0, SOURCE_X_MIN_PT, 415.0),)),
        # q39's right-side ``90`` starts just before its numbered text row.
        "os_q38": (548.0, 621.85, ()),
        # Tree/graph drawings that continue below the next numbered row.
        "ds_q32": (455.3, 531.0, ((515.28, 531.0, SOURCE_X_MIN_PT, 420.0),)),
        "ds_q37": (178.3, 255.0, ((222.34, 255.0, SOURCE_X_MIN_PT, 420.0),)),
        "ds_a05": (173.8, 243.0, ((233.22, 243.0, SOURCE_X_MIN_PT, 380.0),)),
        "ds_a42": (182.8, 295.0, ((243.84, 295.0, SOURCE_X_MIN_PT, 400.0),)),
        "ds_a20": (497.8, 600.0, ((556.76, 600.0, SOURCE_X_MIN_PT, 395.0),)),
        "ds_a18": (
            568.8,
            690.0,
            (
                (568.8, 580.0, SOURCE_X_MIN_PT, 400.0),
                (629.47, 690.0, SOURCE_X_MIN_PT, 400.0),
            ),
        ),
        # The answer-01 graph floats above its answer heading; mask the prior
        # answer's left-column prose while retaining the complete right graph.
        "ds_a21": (
            85.0,
            185.0,
            (
                (64.0, 126.3, SOURCE_X_MIN_PT, 390.0),
                (64.0, 94.0, 390.0, SOURCE_X_MAX_PT),
                (157.26, 185.0, SOURCE_X_MIN_PT, 390.0),
            ),
        ),
        # Answer 02 starts beside the tail of answer 01's graph and ends
        # beside answer 03's heading; keep only its own matrix/graph content.
        "ds_a22": (
            157.3,
            356.0,
            (
                (157.3, 185.0, 390.0, SOURCE_X_MAX_PT),
                (311.26, 356.0, SOURCE_X_MIN_PT, 390.0),
            ),
        ),
        "ds_q49": (174.3, 278.0, ()),
        "ds_q58_tail": (64.0, 162.0, ((151.39, 162.0, SOURCE_X_MIN_PT, 395.0),)),
        "ds_a23_tail": (64.0, 93.0, ()),
        # The remaining regions are trimmed before the next block's graphics.
        "ds_q36": (119.3, 176.0, ()),
        "ds_q12": (454.3, 542.0, ()),
        "ds_q05": (153.3, 250.0, ()),
        "ds_q07": (379.2, 423.5, ()),
        "ds_q15": (590.4, 634.5, ()),
        "ds_q16": (636.4, 705.0, ((681.18, 705.0, SOURCE_X_MIN_PT, 335.0),)),
        "ds_q26": (104.4, 174.5, ()),
        "ds_q21": (180.3, 222.5, ()),
        "co_q49": (102.4, 185.5, ()),
        "co_a20_tail": (64.0, 132.0, ()),
    }
    try:
        top, bottom, masks = rules[special]
    except KeyError as exc:
        raise ValueError(f"unknown special crop: {special}") from exc
    return crop_masked(page, top, bottom, masks)


def block_segments(
    starts: tuple[Start, ...],
    index: int,
    end: Boundary,
    special: Optional[str],
    text_index: Optional[TextIndex] = None,
) -> list[Segment]:
    start = starts[index]
    next_start = (
        starts[index + 1]
        if index + 1 < len(starts)
        else Start(end.page, end.y, -1, -1, end.y_max, end.x, end.x_max, end.text)
    )
    special_segments = {
        "os_q7": [Segment(start.page, 285.0, 386.0, "os_q7")],
        "os_q38": [Segment(start.page, 548.0, 621.85, "os_q38")],
        "ds_q32": [Segment(start.page, 455.3, 531.0, "ds_q32")],
        "ds_q36": [Segment(start.page, 119.3, 176.0, "ds_q36")],
        "ds_q37": [Segment(start.page, 178.3, 255.0, "ds_q37")],
        "ds_a05": [Segment(start.page, 173.8, 243.0, "ds_a05")],
        "ds_a42": [Segment(start.page, 182.8, 295.0, "ds_a42")],
        "ds_q12": [Segment(start.page, 454.3, 542.0, "ds_q12")],
        "ds_a20": [Segment(start.page, 497.8, 600.0, "ds_a20")],
        "ds_a18": [Segment(start.page, 568.8, 690.0, "ds_a18")],
        "ds_a23": [Segment(start.page, start.y, CONTENT_BOTTOM_PT)],
        "ds_q05": [Segment(start.page, 153.3, 250.0, "ds_q05")],
        "ds_a21": [Segment(start.page, 85.0, 185.0, "ds_a21")],
        "ds_a22": [Segment(start.page, 157.3, 356.0, "ds_a22")],
        "ds_q07": [Segment(start.page, 379.2, 423.5, "ds_q07")],
        "ds_q15": [Segment(start.page, 590.4, 634.5, "ds_q15")],
        "ds_q16": [Segment(start.page, 636.4, 705.0, "ds_q16")],
        "ds_q26": [Segment(start.page, 104.4, 174.5, "ds_q26")],
        "ds_q49": [Segment(start.page, 174.3, 278.0, "ds_q49")],
        "ds_q58": [
            Segment(start.page, start.y, CONTENT_BOTTOM_PT),
            Segment(start.page + 1, CONTENT_TOP_PT, 162.0, "ds_q58_tail"),
        ],
        "ds_q21": [Segment(start.page, 180.3, 222.5, "ds_q21")],
        "co_q49": [Segment(start.page, 102.4, 185.5, "co_q49")],
        "co_a20": [
            Segment(start.page, start.y, CONTENT_BOTTOM_PT),
            Segment(start.page + 1, CONTENT_TOP_PT, 132.0, "co_a20_tail"),
        ],
    }
    if special in special_segments:
        return special_segments[special]
    result: list[Segment] = []
    for page in range(start.page, next_start.page + 1):
        top = start.y if page == start.page else CONTENT_TOP_PT
        if page == next_start.page:
            # A neighboring numbered block may have overlapping glyph boxes;
            # the yMax-based extension below keeps its lower line intact.  A
            # section end is different: leave a little more room before the
            # following prose heading so its antialiased top edge is excluded.
            boundary_gap = BOUNDARY_GAP_PT if next_start.ordinal >= 0 else END_BOUNDARY_GAP_PT
            bottom = next_start.y - boundary_gap
        else:
            bottom = CONTENT_BOTTOM_PT
        if text_index is not None and page == next_start.page:
            # OCR yMin values for adjacent lines can overlap by a few points,
            # especially in dense answer pages. Keep the complete lower edge
            # of the current line instead of cutting it at next_start.y.
            line_bottoms = [
                line.y_max
                for line in text_index.page_lines(page)
                if line.y >= top - 0.01 and line.y < next_start.y - 0.01
            ]
            if line_bottoms:
                required_bottom = max(line_bottoms) + BLOCK_BOTTOM_PADDING_PT
                bottom = max(
                    bottom,
                    min(required_bottom, next_start.y + MAX_BOUNDARY_EXTENSION_PT),
                )
        if bottom > top:
            result.append(Segment(page, top, bottom))
    if special == "q36":
        return [Segment(start.page, 344.0, 455.0, "q36")]
    if special == "q37":
        return [Segment(start.page, 410.0, 469.45, "q37")]
    return result


def pad_vertical(page: PPMPage, strip: tuple[int, int, bytes]) -> tuple[int, int, bytes]:
    width, height, data = strip
    padding = round(CROP_PADDING_PT * page.height / SOURCE_HEIGHT_PT)
    if padding <= 0:
        return strip
    row_size = width * 3
    blank = b"\xff" * (width * padding * 3)
    return width, height + 2 * padding, blank + data + blank


def add_horizontal_margin(strip: tuple[int, int, bytes], margin_px: int) -> tuple[int, int, bytes]:
    width, height, data = strip
    row_size = width * 3
    blank = b"\xff" * (margin_px * 3)
    output_row = blank + data[:row_size] + blank
    rows = [output_row]
    for offset in range(row_size, len(data), row_size):
        rows.append(blank + data[offset : offset + row_size] + blank)
    return width + 2 * margin_px, height, b"".join(rows)


def get_strip(
    cache: RenderCache,
    segment: Segment,
    margin_px: int,
) -> tuple[int, int, bytes]:
    page = cache.get(segment.page)
    if segment.special == "q36":
        raw = crop_q36(page)
    elif segment.special == "q37":
        raw = crop_q37(page)
    elif segment.special is not None:
        raw = crop_special(page, segment.special)
    else:
        raw = crop(page, segment.top, segment.bottom)
    return add_horizontal_margin(pad_vertical(page, raw), margin_px)


def get_title_strip(
    cache: RenderCache,
    boundary: Boundary,
    margin_px: int,
) -> tuple[int, int, bytes]:
    page = cache.get(boundary.page)
    raw = crop_title_line(page, boundary)
    return add_horizontal_margin(pad_vertical(page, raw), margin_px)


def blank_strip(width: int, page: PPMPage, points: float) -> tuple[int, int, bytes]:
    height = max(1, round(points * page.height / SOURCE_HEIGHT_PT))
    return width, height, b"\xff" * (width * height * 3)


def safe_split_row(strip: tuple[int, int, bytes], max_rows: int) -> Optional[int]:
    """Return a cut row only when the source strip has a real blank gap."""
    width, height, data = strip
    if max_rows < 1 or max_rows >= height:
        return None
    row_size = width * 3
    min_fragment = max(1, round(18.0 * height / SOURCE_HEIGHT_PT))
    if max_rows < min_fragment * 2:
        return None
    blank_runs: list[tuple[int, int]] = []
    run_start: Optional[int] = None
    # The margin is white by construction.  Ignore it when deciding whether a
    # split is safe, and require a multi-row blank run across the strip.
    for row in range(min(height, max_rows)):
        pixels = data[row * row_size : (row + 1) * row_size]
        dark = 0
        for offset in range(0, len(pixels), 3):
            if max(pixels[offset : offset + 3]) < 220:
                dark += 1
        is_blank = dark <= max(2, width // 600)
        if is_blank and run_start is None:
            run_start = row
        elif not is_blank and run_start is not None:
            if row - run_start >= 4:
                blank_runs.append((run_start, row))
            run_start = None
    if run_start is not None and max_rows - run_start >= 4:
        blank_runs.append((run_start, max_rows))
    candidates = [
        (start + (end - start) // 2)
        for start, end in blank_runs
        if start >= min_fragment and end <= max_rows and height - end >= min_fragment
    ]
    return min(candidates, key=lambda row: abs(row - max_rows)) if candidates else None


def append_pair(
    output_pages: list[bytes],
    current: bytearray,
    width: int,
    height: int,
    strips: Iterable[tuple[int, int, bytes]],
    margin_px: int,
) -> None:
    row_size = width * 3
    content_limit = height - margin_px

    def start_page() -> None:
        if not current:
            current.extend(b"\xff" * (margin_px * row_size))

    def finish_page() -> None:
        if current:
            current.extend(b"\xff" * (height - len(current) // row_size) * row_size)
            output_pages.append(bytes(current))
            current.clear()

    start_page()
    for strip_width, strip_height, strip_data in strips:
        if strip_width != width:
            raise ValueError(f"inconsistent strip width: {strip_width} != {width}")
        remaining = content_limit - len(current) // row_size
        if strip_height > remaining:
            if len(current) // row_size > margin_px:
                cut = safe_split_row((strip_width, strip_height, strip_data), remaining)
                if cut is not None:
                    current.extend(strip_data[: cut * row_size])
                    finish_page()
                    start_page()
                    strip_height -= cut
                    strip_data = strip_data[cut * row_size :]
                    remaining = content_limit - len(current) // row_size
                else:
                    finish_page()
                    start_page()
                    remaining = content_limit - len(current) // row_size
        if strip_height <= remaining:
            current.extend(strip_data)
            continue

        # A normal block is shorter than an output page.  This fallback keeps
        # the renderer usable for a rare oversized block while preserving all
        # pixels instead of silently dropping content.
        offset = 0
        while offset < strip_height:
            remaining = content_limit - len(current) // row_size
            if remaining == 0:
                finish_page()
                start_page()
                remaining = content_limit - len(current) // row_size
            take = min(remaining, strip_height - offset)
            begin = offset * row_size
            end = (offset + take) * row_size
            current.extend(strip_data[begin:end])
            offset += take
            if offset < strip_height and len(current) // row_size == content_limit:
                finish_page()
                start_page()


class PDFStreamWriter:
    """Write image pages incrementally so the full book stays memory bounded."""

    def __init__(self, path: Path, width_pt: float, height_pt: float):
        self.handle = path.open("wb")
        self.handle.write(b"%PDF-1.4\n%\xff\xff\xff\xff\n")
        self.width_pt = width_pt
        self.height_pt = height_pt
        self.offsets: list[Optional[int]] = [None, None, None]
        self.next_id = 3
        self.page_ids: list[int] = []

    def _reserve(self) -> int:
        number = self.next_id
        self.next_id += 1
        self.offsets.append(None)
        return number

    def _write_object(self, number: int, data: bytes) -> None:
        self.offsets[number] = self.handle.tell()
        self.handle.write(f"{number} 0 obj\n".encode())
        self.handle.write(data)
        self.handle.write(b"\nendobj\n")

    def add_page(self, raw: bytes, width_px: int, height_px: int) -> None:
        image_id = self._reserve()
        content_id = self._reserve()
        page_id = self._reserve()
        page_number = len(self.page_ids) + 1
        compressed = zlib.compress(raw, 6)
        image_dict = (
            f"<< /Type /XObject /Subtype /Image /Width {width_px} /Height {height_px} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(compressed)} >>\nstream\n".encode()
            + compressed
            + b"\nendstream"
        )
        self._write_object(image_id, image_dict)
        content = f"q {self.width_pt:g} 0 0 {self.height_pt:g} 0 0 cm /Im{page_number} Do Q".encode()
        self._write_object(content_id, f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
        page = (
            f"<< /Type /Page /Parent 1 0 R /MediaBox [0 0 {self.width_pt:g} {self.height_pt:g}] "
            f"/Resources << /XObject << /Im{page_number} {image_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode()
        self._write_object(page_id, page)
        self.page_ids.append(page_id)

    def close(self) -> None:
        kids = " ".join(f"{page_id} 0 R" for page_id in self.page_ids)
        self._write_object(1, f"<< /Type /Pages /Kids [{kids}] /Count {len(self.page_ids)} >>".encode())
        self._write_object(2, b"<< /Type /Catalog /Pages 1 0 R >>")
        xref = self.handle.tell()
        self.handle.write(f"xref\n0 {self.next_id}\n".encode())
        self.handle.write(b"0000000000 65535 f \n")
        for offset in self.offsets[1:]:
            if offset is None:
                raise ValueError("missing PDF object offset")
            self.handle.write(f"{offset:010d} 00000 n \n".encode())
        self.handle.write(
            f"trailer\n<< /Size {self.next_id} /Root 2 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        )
        self.handle.close()


def special_for(
    config: BookConfig,
    prefix: str,
    question_index: int,
    kind: str = "q",
) -> Optional[str]:
    for region_prefix, region_name, region_index in config.special_regions:
        if prefix != region_prefix or question_index != region_index:
            continue
        if f"_{kind}" in region_name or region_name.startswith(f"{kind}_") or (
            kind == "q" and region_name in ("q36", "q37")
        ):
            return region_name
    return None


def build_pdf(
    config: BookConfig,
    sections: list[ExerciseSection],
    text_index: TextIndex,
    cache: RenderCache,
    output: Path,
    *,
    preview_pairs: int = 0,
) -> tuple[int, int, int, float]:
    first_page = cache.get(sections[0].question_starts[0].page)
    source_x0, _ = point_to_px(first_page, SOURCE_X_MIN_PT, 0)
    source_x1, _ = point_to_px(first_page, SOURCE_X_MAX_PT, 0)
    content_width_px = source_x1 - source_x0
    margin_px = round(PAGE_MARGIN_PT * first_page.height / SOURCE_HEIGHT_PT)
    output_width_px = content_width_px + 2 * margin_px
    output_height_px = first_page.height
    output_width_pt = output_width_px * SOURCE_WIDTH_PT / first_page.width
    writer = PDFStreamWriter(output, output_width_pt, SOURCE_HEIGHT_PT)
    current = bytearray()
    output_page_count = 0
    pair_count = 0
    selected_sections = sections

    for section_index, section in enumerate(selected_sections):
        starts_q = section.question_starts
        starts_a = section.answer_starts
        for pair_index, (question, answer) in enumerate(zip(starts_q, starts_a)):
            if preview_pairs and pair_count >= preview_pairs:
                break
            q_special = special_for(config, section.prefix, pair_index, "q")
            q_segments = block_segments(
                starts_q,
                pair_index,
                section.question_end,
                q_special,
                text_index,
            )
            a_special = special_for(config, section.prefix, pair_index, "a")
            a_segments = block_segments(
                starts_a,
                pair_index,
                section.answer_end,
                a_special,
                text_index,
            )
            q_strips = [get_strip(cache, item, margin_px) for item in q_segments]
            a_strips = [get_strip(cache, item, margin_px) for item in a_segments]
            strips: list[tuple[int, int, bytes]] = []
            if pair_index == 0:
                # Keep source-rendered chapter and subsection labels immediately
                # before the first question of each exercise subsection.
                strips.append(get_title_strip(cache, section.chapter_title, margin_px))
                strips.append(
                    blank_strip(
                        output_width_px,
                        cache.get(section.chapter_title.page),
                        CHAPTER_SECTION_GAP_PT,
                    )
                )
                strips.append(get_title_strip(cache, section.section_title, margin_px))
                strips.append(
                    blank_strip(
                        output_width_px,
                        cache.get(section.section_title.page),
                        SECTION_QUESTION_GAP_PT,
                    )
                )
            strips.extend(q_strips)
            strips.append(blank_strip(output_width_px, cache.get(question.page), QUESTION_ANSWER_GAP_PT))
            strips.extend(a_strips)
            if pair_index + 1 < len(starts_q):
                strips.append(blank_strip(output_width_px, cache.get(answer.page), ANSWER_QUESTION_GAP_PT))
            completed: list[bytes] = []
            # Completed output pages are streamed immediately, so the whole
            # book is never retained as raw RGB data in memory.
            append_pair(completed, current, output_width_px, output_height_px, strips, margin_px)
            for raw in completed:
                writer.add_page(raw, output_width_px, output_height_px)
                output_page_count += 1
            pair_count += 1
        if preview_pairs and pair_count >= preview_pairs:
            break

    if current:
        row_size = output_width_px * 3
        current.extend(b"\xff" * ((output_height_px - len(current) // row_size) * row_size))
        writer.add_page(bytes(current), output_width_px, output_height_px)
        output_page_count += 1
    writer.close()
    return pair_count, output_page_count, output_width_px, output_width_pt


def write_report(
    path: Path,
    config: BookConfig,
    sections: list[ExerciseSection],
    pair_count: int,
    page_count: int,
    width_pt: float,
    dpi: int,
) -> None:
    lines = [
        f"{config.label} exercise reflow QA inventory",
        f"book={config.key}",
        f"dpi={dpi}",
        f"output_width_pt={width_pt:.3f}",
        f"source_crop_x_pt={SOURCE_X_MIN_PT:.1f}..{SOURCE_X_MAX_PT:.1f}",
        f"content_y_pt={CONTENT_TOP_PT:.1f}..{CONTENT_BOTTOM_PT:.1f}",
        f"page_margin_pt={PAGE_MARGIN_PT:.1f}",
        f"crop_padding_pt={CROP_PADDING_PT:.1f}",
        f"block_bottom_padding_pt={BLOCK_BOTTOM_PADDING_PT:.2f}",
        f"end_boundary_gap_pt={END_BOUNDARY_GAP_PT:.1f}",
        f"max_boundary_extension_pt={MAX_BOUNDARY_EXTENSION_PT:.1f}",
        f"question_answer_gap_pt={QUESTION_ANSWER_GAP_PT:.1f}",
        f"answer_question_gap_pt={ANSWER_QUESTION_GAP_PT:.1f}",
        f"chapter_section_gap_pt={CHAPTER_SECTION_GAP_PT:.1f}",
        f"section_question_gap_pt={SECTION_QUESTION_GAP_PT:.1f}",
        "locator_titles=source-rendered chapter running header + subsection heading before first question",
        "special_regions="
        + ("; ".join(f"{prefix}/{name}/q{index + 1}" for prefix, name, index in config.special_regions) or "none"),
        f"sections={len(sections)}",
        f"pairs={pair_count}",
        f"output_pages={page_count}",
        "",
    ]
    for section in sections:
        q_end = section.question_end
        a_end = section.answer_end
        end_text = a_end.text.replace("\n", " ").strip()
        lines.append(
            f"{section.prefix}: {len(section.question_starts)} pairs; "
            f"chapter_title=p{section.chapter_title.page}@{section.chapter_title.y:.2f}; "
            f"section_title=p{section.section_title.page}@{section.section_title.y:.2f}; "
            f"question_end=p{q_end.page}@{q_end.y:.2f}; "
            f"answer_end=p{a_end.page}@{a_end.y:.2f} {end_text}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", choices=sorted(BOOK_CONFIGS), default="network")
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--preview-pairs", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.dpi < 120:
        raise ValueError("dpi is too low for a study copy")
    config = BOOK_CONFIGS[args.book]

    with tempfile.TemporaryDirectory(prefix=f"{config.key}-reflow-") as temp_name:
        temp_dir = Path(temp_name)
        index = build_text_index(args.source_pdf, temp_dir)
        sections = build_sections(index, config)
        print("inventory:", ", ".join(f"{s.prefix}={len(s.question_starts)}" for s in sections))
        cache = RenderCache(args.source_pdf, args.dpi, temp_dir)
        pair_count, page_count, _, width_pt = build_pdf(
            config,
            sections,
            index,
            cache,
            args.output,
            preview_pairs=args.preview_pairs,
        )
        report = args.report or args.output.with_suffix(".qa.txt")
        write_report(report, config, sections, pair_count, page_count, width_pt, args.dpi)
    print(f"wrote {args.output}: {pair_count} pairs, {page_count} pages")
    print(f"qa report: {report}")


if __name__ == "__main__":
    main()
