"""IEEE-style two-column PDF typesetter (reportlab) for the conference paper."""
import os, sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, Table, TableStyle, FrameBreak,
                                NextPageTemplate, KeepTogether, CondPageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE_W, PAGE_H = letter
M_LEFT = M_RIGHT = 0.62 * inch
M_TOP = 0.62 * inch
M_BOT = 0.60 * inch
GUTTER = 0.24 * inch
COL_W = (PAGE_W - M_LEFT - M_RIGHT - GUTTER) / 2.0

FONT_DIR = r"C:\Windows\Fonts"
try:
    pdfmetrics.registerFont(TTFont("Times", os.path.join(FONT_DIR, "times.ttf")))
    pdfmetrics.registerFont(TTFont("Times-Bold", os.path.join(FONT_DIR, "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("Times-Italic", os.path.join(FONT_DIR, "timesi.ttf")))
    pdfmetrics.registerFont(TTFont("Times-BoldItalic", os.path.join(FONT_DIR, "timesbi.ttf")))
    BODY_FONT = "Times"
except Exception:
    BODY_FONT = "Times-Roman"

S = {}
S["body"] = ParagraphStyle(
    "body", fontName=BODY_FONT, fontSize=10.0, leading=12.0, alignment=TA_JUSTIFY,
    firstLineIndent=10.5, spaceBefore=0, spaceAfter=0,
    hyphenationLang="en_US", embeddedHyphenation=1, uriWasteReduce=0.3)
S["body_ni"] = ParagraphStyle("body_ni", parent=S["body"], firstLineIndent=0)
S["abstract"] = ParagraphStyle(
    "abstract", parent=S["body"], fontSize=10.0, leading=12.0, leftIndent=0, rightIndent=0,
    firstLineIndent=0, spaceAfter=2)
S["title"] = ParagraphStyle("title", fontName=BODY_FONT, fontSize=20.5, leading=24,
                            alignment=TA_CENTER, spaceBefore=0, spaceAfter=6)
S["authors"] = ParagraphStyle("authors", fontName=BODY_FONT, fontSize=10.5, leading=13,
                              alignment=TA_CENTER, spaceAfter=3)
S["affil"] = ParagraphStyle("affil", fontName="Times-Italic", fontSize=9, leading=11,
                            alignment=TA_CENTER, spaceAfter=10)
S["h1"] = ParagraphStyle("h1", fontName=BODY_FONT, fontSize=10.5, leading=12.5,
                         spaceBefore=7, spaceAfter=3.5, alignment=TA_CENTER,
                         textTransform="uppercase")
S["h2"] = ParagraphStyle("h2", fontName="Times-Italic", fontSize=10.0, leading=12.0,
                         spaceBefore=4.5, spaceAfter=2.5, alignment=TA_LEFT)
S["caption"] = ParagraphStyle("caption", fontName=BODY_FONT, fontSize=9.0, leading=10.6,
                              alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=4)
S["table_cap"] = ParagraphStyle("table_cap", fontName=BODY_FONT, fontSize=9.0, leading=10.6,
                                alignment=TA_CENTER, spaceBefore=1, spaceAfter=2)
S["table_head_cap"] = ParagraphStyle("table_head_cap", fontName=BODY_FONT, fontSize=9.0,
                                     leading=10.6, alignment=TA_CENTER, spaceBefore=4,
                                     spaceAfter=1)
S["ref"] = ParagraphStyle("ref", fontName=BODY_FONT, fontSize=7.4, leading=8.0,
                          alignment=TA_JUSTIFY, leftIndent=13, firstLineIndent=-13,
                          spaceAfter=0.2)
S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=12, firstLineIndent=0,
                             spaceAfter=1.5)
S["table_cell"] = ParagraphStyle("cell", fontName=BODY_FONT, fontSize=7.8, leading=9.0,
                                 alignment=TA_CENTER)
S["table_cell_l"] = ParagraphStyle("cell_l", parent=S["table_cell"], alignment=TA_LEFT)
S["table_head"] = ParagraphStyle("th", parent=S["table_cell"], fontName=BODY_FONT)


def first_page(canvas, doc):
    # IEEE conference camera-ready carries no page numbers
    return


def later_pages(canvas, doc):
    return


def eq_png(tex, name, fontsize=9.2):
    """Render a LaTeX-ish equation with matplotlib mathtext to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, "$" + tex + "$", fontsize=fontsize)
    path = os.path.join(BASE, "figures", f"eq_{name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    return path


def eq(tex, name, width_pt=None, height_pt=None):
    path = eq_png(tex, name)
    from PIL import Image as PImage
    iw, ih = PImage.open(path).size
    h = height_pt or 14
    w = h * iw / ih
    max_w = COL_W * 0.88
    if w > max_w:
        w = max_w
        h = max_w * ih / iw
    if width_pt:
        w = width_pt
        h = width_pt * ih / iw
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    num = Paragraph("(1)", ParagraphStyle("eqnum", parent=S["body"], alignment=TA_RIGHT,
                                          fontSize=8.6))
    t = Table([[img, num]], colWidths=[COL_W - 16, 16], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def figure(path, caption, width_frac=1.0, space_before=4):
    img = Image(path, width=COL_W * width_frac,
                height=COL_W * width_frac * _aspect(path))
    img.hAlign = "CENTER"
    cap = Paragraph(caption, S["caption"])
    return [Spacer(1, space_before), img, cap]


def _aspect(path):
    from PIL import Image as PImage
    iw, ih = PImage.open(path).size
    return ih / iw


def table(data, caption, col_widths=None, align_left_col0=True, font_size=7.35,
          row_height=None):
    import re as _re
    m = _re.match(r"^\s*<b>Table\s+([IVX]+)\.</b>\s*(.*)$", caption, _re.S)
    if m:
        head = Paragraph(f"TABLE {m.group(1)}", S["table_head_cap"])
        body = Paragraph(m.group(2), S["table_cap"])
        cap = [head, body]
    else:
        cap = [Paragraph(caption, S["table_cap"])]
    t = Table(data, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), BODY_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.3),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
    ]
    if align_left_col0:
        style.append(("ALIGN", (0, 0), (0, -1), "LEFT"))
    t.setStyle(TableStyle(style))
    return [KeepTogether(cap + [t]), Spacer(1, 6)]


def _height(f, width):
    w, h = f.wrap(width, PAGE_H)
    st = getattr(f, "style", None)
    sa = getattr(st, "spaceBefore", 0) or 0
    sb = getattr(st, "spaceAfter", 0) or 0
    return h + sa + sb


def build(blocks, out_path, title_meta="Interpretable Prediction of Student Academic Performance"):
    # measure title block height
    title_flow = []
    for b in blocks:
        if b[0] == "title_end":
            break
        title_flow.append(_flowable(b))
    h_title = 0
    for f in title_flow:
        if isinstance(f, list):
            for sub in f:
                h_title += _height(sub, PAGE_W - 2 * M_LEFT)
        else:
            h_title += _height(f, PAGE_W - 2 * M_LEFT)
    h_title += 6

    doc = BaseDocTemplate(out_path, pagesize=letter,
                          leftMargin=M_LEFT, rightMargin=M_RIGHT,
                          topMargin=M_TOP, bottomMargin=M_BOT,
                          title=title_meta, author="Anonymous submission",
                          subject="Learning analytics; interpretability")
    body_h = PAGE_H - M_TOP - M_BOT
    frames_first = [
        Frame(M_LEFT, PAGE_H - M_TOP - h_title, PAGE_W - M_LEFT - M_RIGHT, h_title,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="title"),
        Frame(M_LEFT, M_BOT, COL_W, body_h - h_title, leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0, id="l1"),
        Frame(M_LEFT + COL_W + GUTTER, M_BOT, COL_W, body_h - h_title,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="r1"),
    ]
    frames_rest = [
        Frame(M_LEFT, M_BOT, COL_W, body_h, leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0, id="l"),
        Frame(M_LEFT + COL_W + GUTTER, M_BOT, COL_W, body_h, leftPadding=0,
              rightPadding=0, topPadding=0, bottomPadding=0, id="r"),
    ]
    doc.addPageTemplates([
        PageTemplate(id="first", frames=frames_first, onPage=first_page),
        PageTemplate(id="rest", frames=frames_rest, onPage=later_pages),
    ])

    story = [NextPageTemplate("rest")]
    i = 0
    for b in blocks:
        if b[0] == "title_end":
            story.append(FrameBreak())
            i += 1
            break
        f = _flowable(b)
        story.extend(f if isinstance(f, list) else [f])
        i += 1
    for b in blocks[i:]:
        f = _flowable(b)
        if b[0] == "h1":
            story.append(CondPageBreak(0.9 * inch))
        story.extend(f if isinstance(f, list) else [f])
    doc.build(story)
    print("built", out_path)


def _flowable(b):
    kind = b[0]
    if kind == "title":
        return Paragraph(b[1], S["title"])
    if kind == "authors":
        return Paragraph(b[1], S["authors"])
    if kind == "affil":
        return Paragraph(b[1], S["affil"])
    if kind == "h1":
        return Paragraph(b[1], S["h1"])
    if kind == "h2":
        return Paragraph(b[1], S["h2"])
    if kind == "p":
        return Paragraph(b[1], S["body"])
    if kind == "bullet":
        st = ParagraphStyle("bullet", parent=S["body"], leftIndent=9, firstLineIndent=-7)
        return Paragraph(b[1], st)
    if kind == "pni":
        return Paragraph(b[1], S["body_ni"])
    if kind == "abs":
        return Paragraph(b[1], S["abstract"])
    if kind == "ref":
        return Paragraph(b[1], S["ref"])
    if kind == "space":
        return Spacer(1, b[1])
    if kind == "fig":
        return figure(os.path.join(BASE, "figures", b[1]), b[2],
                      width_frac=b[3] if len(b) > 3 else 1.0)
    if kind == "tab":
        return table(b[1], b[2], col_widths=b[3] if len(b) > 3 else None,
                     font_size=b[4] if len(b) > 4 else 7.35)
    if kind == "eq":
        return [Spacer(1, 4), eq(b[1], b[2], height_pt=b[3] if len(b) > 3 else 17), Spacer(1, 5)]
    raise ValueError(kind)
