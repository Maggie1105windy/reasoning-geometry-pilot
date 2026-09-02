"""Render the reviewed Stage B Markdown report as a polished Chinese PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "stage-b" / "STAGE_B_RESEARCH_REPORT_ZH.md"
OUTPUT = ROOT / "output" / "pdf" / "STAGE_B_RESEARCH_REPORT_ZH.pdf"

FONT_DIR = Path(r"C:\Windows\Fonts")
BODY_FONT = "DengXian"
BODY_BOLD = "DengXian-Bold"
HEADING_FONT = "SimHei"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#167C80")
LIGHT_TEAL = colors.HexColor("#E9F4F3")
LIGHT_BLUE = colors.HexColor("#EDF3F8")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5C6B78")
LINE = colors.HexColor("#C9D6DF")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(BODY_FONT, str(FONT_DIR / "Deng.ttf")))
    pdfmetrics.registerFont(TTFont(BODY_BOLD, str(FONT_DIR / "Dengb.ttf")))
    pdfmetrics.registerFont(TTFont(HEADING_FONT, str(FONT_DIR / "simhei.ttf")))
    pdfmetrics.registerFontFamily(
        BODY_FONT,
        normal=BODY_FONT,
        bold=BODY_BOLD,
        italic=BODY_FONT,
        boldItalic=BODY_BOLD,
    )


def normalize_dashes(value: str) -> str:
    return (
        value.replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )


def inline_markup(value: str) -> str:
    value = normalize_dashes(value.strip())
    tokens: list[str] = []

    def stash(fragment: str) -> str:
        tokens.append(fragment)
        return f"@@TOKEN{len(tokens) - 1}@@"

    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda m: stash(
            f'<link href="{html.escape(m.group(2), quote=True)}" color="#167C80">'
            f"{html.escape(m.group(1))}</link>"
        ),
        value,
    )
    value = re.sub(
        r"`([^`]+)`",
        lambda m: stash(
            f'<font name="{BODY_BOLD}" color="#17324D">{html.escape(m.group(1))}</font>'
        ),
        value,
    )
    value = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: stash(f"<b>{html.escape(m.group(1))}</b>"),
        value,
    )
    value = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        lambda m: stash(f"<i>{html.escape(m.group(1))}</i>"),
        value,
    )
    value = html.escape(value)
    value = re.sub(
        r"(?<![\"'=])(https?://[^\s<]+)",
        lambda m: stash(
            f'<link href="{html.escape(m.group(1), quote=True)}" color="#167C80">'
            f"{html.escape(m.group(1))}</link>"
        ),
        value,
    )
    for index, fragment in enumerate(tokens):
        value = value.replace(f"@@TOKEN{index}@@", fragment)
    return value


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=sample["Title"],
            fontName=HEADING_FONT,
            fontSize=25,
            leading=36,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            fontName=BODY_FONT,
            fontSize=11,
            leading=18,
            textColor=MUTED,
            spaceAfter=6 * mm,
        ),
        "cover_label": ParagraphStyle(
            "CoverLabel",
            fontName=BODY_BOLD,
            fontSize=10,
            leading=16,
            textColor=NAVY,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle",
            fontName=HEADING_FONT,
            fontSize=22,
            leading=28,
            textColor=NAVY,
            spaceAfter=8 * mm,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            fontName=HEADING_FONT,
            fontSize=17,
            leading=24,
            textColor=NAVY,
            spaceBefore=8 * mm,
            spaceAfter=3.5 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            fontName=BODY_BOLD,
            fontSize=12.5,
            leading=19,
            textColor=TEAL,
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=BODY_FONT,
            fontSize=9.4,
            leading=15,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=2.8 * mm,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=BODY_FONT,
            fontSize=9.2,
            leading=14.5,
            textColor=INK,
            leftIndent=6 * mm,
            firstLineIndent=-4 * mm,
            spaceAfter=1.4 * mm,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName=BODY_FONT,
            fontSize=8.2,
            leading=12.5,
            textColor=MUTED,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "Table",
            fontName=BODY_FONT,
            fontSize=7.5,
            leading=10.5,
            textColor=INK,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            fontName=BODY_BOLD,
            fontSize=7.5,
            leading=10.5,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


class ResearchDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, styles: dict[str, ParagraphStyle]) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=21 * mm,
            bottomMargin=20 * mm,
            title="Stage B 研究报告",
            author="Reasoning Geometry Pilot",
            subject="StepGame mechanistic probing and post-training comparison",
        )
        self.styles = styles
        cover_frame = Frame(
            18 * mm,
            18 * mm,
            A4[0] - 36 * mm,
            A4[1] - 36 * mm,
            id="cover-frame",
        )
        body_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body-frame",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[cover_frame], onPage=self._cover_page),
                PageTemplate(id="body", frames=[body_frame], onPage=self._body_page),
            ]
        )

    @staticmethod
    def _cover_page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, A4[1] - 20 * mm, A4[0], 20 * mm, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 0, A4[0], 7 * mm, fill=1, stroke=0)
        canvas.restoreState()

    @staticmethod
    def _body_page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, A4[1] - 13 * mm, A4[0] - 18 * mm, A4[1] - 13 * mm)
        canvas.setFont(BODY_FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, A4[1] - 10.2 * mm, "Stage B 研究报告")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name not in {"Heading1", "Heading2"}:
            return
        level = 0 if flowable.style.name == "Heading1" else 1
        text = flowable.getPlainText()
        key = f"heading-{self.seq.nextf('heading')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def figure_flowable(path: Path, alt: str, number: int, styles, max_width: float):
    with PILImage.open(path) as source:
        width_px, height_px = source.size
    scale = min(max_width / width_px, (112 * mm) / height_px)
    rendered = Image(str(path), width=width_px * scale, height=height_px * scale)
    caption = Paragraph(
        f"图 {number}　{inline_markup(alt)}",
        styles["caption"],
    )
    panel = Table([[rendered], [caption]], colWidths=[max_width])
    panel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.55, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("TOPPADDING", (0, 1), (-1, 1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ]
        )
    )
    return KeepTogether([Spacer(1, 2 * mm), panel, Spacer(1, 3 * mm)])


def table_flowable(rows: list[list[str]], styles, max_width: float):
    columns = len(rows[0])
    converted = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        converted.append([Paragraph(inline_markup(cell), style) for cell in row])

    weights = []
    for col_index in range(columns):
        max_chars = max(len(row[col_index]) if col_index < len(row) else 0 for row in rows)
        weights.append(min(max(max_chars, 8), 30))
    total = sum(weights)
    widths = [max_width * weight / total for weight in weights]

    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(converted)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_BLUE))
    table.setStyle(TableStyle(commands))
    return [Spacer(1, 1.5 * mm), table, Spacer(1, 3 * mm)]


def markdown_story(markdown: str, styles, document_width: float):
    lines = markdown.splitlines()
    story = []
    figure_number = 0
    index = 1  # H1 title is represented on the cover.
    major_section_count = 0

    while index < len(lines):
        raw = lines[index].rstrip()
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            figure_number += 1
            image_path = (SOURCE.parent / image_match.group(2)).resolve()
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            story.append(
                figure_flowable(
                    image_path,
                    image_match.group(1),
                    figure_number,
                    styles,
                    document_width,
                )
            )
            index += 1
            continue

        if stripped.startswith("## "):
            major_section_count += 1
            if major_section_count > 1:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(stripped[3:]), styles["h1"]))
            index += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(inline_markup(stripped[4:]), styles["h2"]))
            index += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [
                [cell.strip() for cell in line.strip("|").split("|")]
                for line in table_lines
            ]
            rows = [
                row
                for row in rows
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)
            ]
            story.extend(table_flowable(rows, styles, document_width))
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        number_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if bullet_match:
            story.append(
                Paragraph(f"<b>•</b>　{inline_markup(bullet_match.group(1))}", styles["bullet"])
            )
            index += 1
            continue
        if number_match:
            story.append(
                Paragraph(
                    f"<b>{number_match.group(1)}.</b>　{inline_markup(number_match.group(2))}",
                    styles["bullet"],
                )
            )
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                candidate.startswith(("## ", "### ", "|", "!["))
                or re.match(r"^[-*]\s+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph_lines)), styles["body"]))

    return story


def build_pdf() -> None:
    register_fonts()
    styles = build_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    markdown = SOURCE.read_text(encoding="utf-8")
    cover_title = (
        "Stage B 研究报告<br/>"
        "后训练增强显式关系读出，<br/>"
        "但未形成稳定的多步推理提升"
    )

    doc = ResearchDocTemplate(str(OUTPUT), styles)
    story = [
        Spacer(1, 42 * mm),
        Paragraph(cover_title, styles["cover_title"]),
        Paragraph(
            "StepGame 空间关系表征、模型行为与 Base/SFT/DPO/Final 后训练阶段比较",
            styles["cover_subtitle"],
        ),
        Table(
            [
                [Paragraph("完成范围", styles["cover_label"]), Paragraph("Stage B0、B1、B1.5、B1.6、B2", styles["body"])],
                [Paragraph("研究对象", styles["cover_label"]), Paragraph("Llama-3.1-8B Base 与 Tülu-3 后训练阶段", styles["body"])],
                [Paragraph("报告日期", styles["cover_label"]), Paragraph("2026-09-02", styles["body"])],
                [Paragraph("验证状态", styles["cover_label"]), Paragraph("B2 final validation: PASS", styles["body"])],
            ],
            colWidths=[30 * mm, 120 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT_TEAL),
                    ("BOX", (0, 0), (-1, -1), 0.6, TEAL),
                    ("INNERGRID", (0, 0), (-1, -1), 0.3, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        ),
        Spacer(1, 28 * mm),
        Paragraph("Reasoning Geometry Pilot · Reviewed research artifact", styles["cover_subtitle"]),
        NextPageTemplate("body"),
        PageBreak(),
        Paragraph("目录", styles["toc_title"]),
    ]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1",
            fontName=BODY_BOLD,
            fontSize=10,
            leading=17,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=3,
        ),
        ParagraphStyle(
            "TOCLevel2",
            fontName=BODY_FONT,
            fontSize=8.5,
            leading=14,
            leftIndent=8 * mm,
            firstLineIndent=0,
            textColor=MUTED,
        ),
    ]
    story.extend([toc, PageBreak()])
    story.extend(markdown_story(markdown, styles, doc.width))
    doc.multiBuild(story)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
