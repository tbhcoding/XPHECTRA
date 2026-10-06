"""
Render a manuscript markdown file as a thesis-formatted PDF.

Committed so that anyone can regenerate the PDFs. They were previously
buildable on one machine only, which made the committed PDFs a single
point of failure: an edit to a .md would leave the .pdf people actually
read showing the old text.

Usage:
    python build_manuscript_pdf.py --in manuscript/chapters_3_to_5.md
    python build_manuscript_pdf.py --in manuscript/chapters_1_to_2.md
    python build_manuscript_pdf.py --in manuscript/DEFENSE_CHEATSHEET.md


Letter page, one-inch margins, Times 12pt, 1.5 line spacing, justified body,
centred chapter headings on fresh pages, gridded tables, page numbers.

Unicode superscripts are converted to ReportLab <super> markup: the built-in
Type 1 fonts have no glyphs for them and would render as black boxes.
"""
import io
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, KeepTogether, Image)
from reportlab.lib.utils import ImageReader
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

BODY, HEAD = "Times-Roman", "Times-Bold"
ITAL = "Times-Italic"

# --------------------------------------------------------------- styles
def S(name, **kw):
    base = dict(fontName=BODY, fontSize=12, leading=18, spaceAfter=0)
    base.update(kw)
    return ParagraphStyle(name, **base)

ST = {
    "chapter": S("chapter", fontName=HEAD, fontSize=14, leading=20,
                 alignment=TA_CENTER, spaceAfter=4),
    "chaptersub": S("chaptersub", fontName=HEAD, fontSize=14, leading=20,
                    alignment=TA_CENTER, spaceAfter=22),
    "h2": S("h2", fontName=HEAD, fontSize=12, leading=18,
            spaceBefore=16, spaceAfter=6),
    "h3": S("h3", fontName=ITAL, fontSize=12, leading=18,
            spaceBefore=12, spaceAfter=4),
    "h4": S("h4", fontName=ITAL, fontSize=11.5, leading=17,
            spaceBefore=10, spaceAfter=3, leftIndent=14),
    "eqn": S("eqn", fontSize=12, leading=18, alignment=TA_CENTER,
             spaceBefore=12, spaceAfter=12),
    "body": S("body", alignment=TA_JUSTIFY, spaceAfter=10, firstLineIndent=0),
    "caption": S("caption", fontSize=11, leading=16, alignment=TA_CENTER,
                 spaceBefore=6, spaceAfter=14),
    "cell": S("cell", fontSize=10, leading=13),
    "cellh": S("cellh", fontName=HEAD, fontSize=10, leading=13),
    "bullet": S("bullet", alignment=TA_JUSTIFY, spaceAfter=7,
                leftIndent=22, bulletIndent=8),
}

# --------------------------------------------------------------- text fixes
SUPER = {"\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
         "\u207b": "-"}

def fix_chars(s):
    """Unicode superscripts -> <super> markup; typographic minus -> hyphen."""
    s = s.replace("\u2212", "-")
    out, run = [], []
    for ch in s:
        if ch in SUPER:
            run.append(SUPER[ch])
        else:
            if run:
                out.append("<super>" + "".join(run) + "</super>")
                run = []
            out.append(ch)
    if run:
        out.append("<super>" + "".join(run) + "</super>")
    return "".join(out)


def inline(s):
    """Markdown inline -> ReportLab markup. Escape first, then re-add tags."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = fix_chars(s)
    s = s.replace("&lt;super&gt;", "<super>").replace("&lt;/super&gt;", "</super>")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier" size="10.5">\1</font>', s)
    return s


# --------------------------------------------------------------- table
def build_table(rows):
    head = [Paragraph(inline(c), ST["cellh"]) for c in rows[0]]
    body = [[Paragraph(inline(c), ST["cell"]) for c in r] for r in rows[1:]]
    ncol = len(rows[0])
    avail = 6.5 * inch
    # give the first column more room; it usually holds the label
    if ncol == 2:
        widths = [avail * 0.45, avail * 0.55]
    else:
        first = avail * (0.34 if ncol <= 4 else 0.28)
        widths = [first] + [(avail - first) / (ncol - 1)] * (ncol - 1)
    t = Table([head] + body, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(.92, .92, .92)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


# --------------------------------------------------------------- parse
def parse(md):
    story, lines, i = [], md.split("\n"), 0
    pending_chapter = None

    while i < len(lines):
        ln = lines[i].rstrip()

        if not ln.strip():
            i += 1
            continue

        # horizontal rule -> chapter break
        if re.fullmatch(r"-{3,}", ln.strip()):
            story.append(PageBreak())
            i += 1
            continue

        # headings
        if ln.startswith("#### "):
            story.append(Paragraph(inline(ln[5:]), ST["h4"]))
            i += 1
            continue
        if ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), ST["h3"]))
            i += 1
            continue
        if ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), ST["h2"]))
            i += 1
            continue
        if ln.startswith("# "):
            txt = ln[2:].strip()
            if pending_chapter is None:
                pending_chapter = txt
                if story:
                    story.append(PageBreak())
                story.append(Paragraph(inline(txt), ST["chapter"]))
            else:
                story.append(Paragraph(inline(txt), ST["chaptersub"]))
                pending_chapter = None
            i += 1
            continue

        # UNMATCHED HEADING GUARD: never leave i unchanged
        if ln.lstrip().startswith("#"):
            story.append(Paragraph(inline(ln.lstrip("# ")), ST["h4"]))
            i += 1
            continue

        # table
        if ln.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [split_row(b) for b in block
                    if not re.fullmatch(r"[\s|:\-]+", b.strip())]
            if rows:
                story.append(Spacer(1, 4))
                story.append(build_table(rows))
                story.append(Spacer(1, 14))
            continue

        # bullet
        if ln.lstrip().startswith(("- ", "* ")):
            story.append(Paragraph(inline(ln.lstrip()[2:]), ST["bullet"],
                                   bulletText="\u2022"))
            i += 1
            continue

        # equation line (markdown blockquote)
        if ln.lstrip().startswith("> "):
            story.append(Paragraph(inline(ln.lstrip()[2:]), ST["eqn"]))
            i += 1
            continue

        # image line
        mimg = re.match(r"!\[(.*?)\]\((.*?)\)", ln.strip())
        if mimg:
            path = mimg.group(2)
            full = os.path.join(ROOT, path) if path else ""
            story.append(Spacer(1, 8))
            if full and os.path.exists(full):
                iw, ih = ImageReader(full).getSize()
                sc = min(6.3 * inch / iw, 4.3 * inch / ih)
                story.append(Image(full, iw * sc, ih * sc, hAlign="CENTER"))
            else:
                ph = Table([[Paragraph(
                    "<i>Figure to be inserted. Export this schematic from the "
                    "existing draft and place it here.</i>", ST["cell"])]],
                    colWidths=[6.3 * inch], rowHeights=[0.95 * inch])
                ph.setStyle(TableStyle([
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14)]))
                story.append(ph)
            story.append(Spacer(1, 4))
            i += 1
            continue

        # figure / table caption line
        if re.match(r"\*\*(Figure|Table) ", ln.strip()):
            story.append(Paragraph(inline(ln.strip()), ST["caption"]))
            i += 1
            continue

        # paragraph: gather until blank
        buf = []
        while i < len(lines) and lines[i].strip() and \
                not lines[i].lstrip().startswith(("|", "- ", "* ", "#", "![", "> ")) and \
                not re.fullmatch(r"-{3,}", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            story.append(Paragraph(inline(" ".join(buf)), ST["body"]))
    return story


# --------------------------------------------------------------- page furniture
def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 11)
    canvas.drawCentredString(letter[0] / 2.0, 0.6 * inch, str(doc.page))
    canvas.restoreState()


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Render a manuscript markdown file as a thesis-formatted PDF.")
    ap.add_argument("--in", dest="src", required=True,
                    help="markdown file to render")
    ap.add_argument("--out", dest="out", default=None,
                    help="output PDF (default: same name, .pdf)")
    a = ap.parse_args()
    out = a.out or a.src.rsplit(".", 1)[0] + ".pdf"
    md = io.open(a.src, encoding="utf-8").read()
    doc = SimpleDocTemplate(
        out, pagesize=letter,
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
        title="LIGTAS-pH Chapters 3-5", author="XPHECTRA")
    story = parse(md)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
