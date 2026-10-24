"""IEEE-conference layout for python-docx: A4, two-column body, Times New Roman.

Used by ``build_paper.py``. Kept separate so the content file reads as a paper, not as XML.

Why each choice:
* Margins and column gap follow the IEEE A4 conference template (top 19 mm, bottom 43 mm,
  sides 14.32 mm, gap 4.22 mm), so the page looks like the official Word template.
* A wide table or figure is placed between two *continuous* section breaks with one column. Word
  then balances the two columns above it, which is how wide floats are done in the Word template.
* Inline markup is a tiny subset (``*italic*``, ``**bold**``, ``_{sub}``, ``^{sup}``) so the content
  can be written as plain strings and still get italics and subscripts.
"""

from __future__ import annotations

import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

FONT = "Times New Roman"
PAGE_W, PAGE_H = Mm(210), Mm(297)
MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDE = Mm(19), Mm(43), Mm(14.32)
COL_GAP = Mm(4.22)
TEXT_W = PAGE_W - 2 * MARGIN_SIDE
COL_W = int((TEXT_W - COL_GAP) / 2)

_TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")


def _font(run, size=None, bold=None, italic=None, smallcaps=None, color=None):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if smallcaps is not None:
        run.font.small_caps = smallcaps
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def add_runs(par, text, size=None, bold=False, italic=False, smallcaps=False):
    """Append ``text`` to ``par``, honouring the inline markup."""
    for part in _TOKEN.split(text):
        if not part:
            continue
        b, i, sub, sup, t = bold, italic, False, False, part
        if part.startswith("**") and part.endswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            i, t = (not italic), part[1:-1]
        elif part.startswith("_{"):
            sub, t = True, part[2:-1]
        elif part.startswith("^{"):
            sup, t = True, part[2:-1]
        run = par.add_run(t)
        _font(run, size=size, bold=b, italic=i, smallcaps=smallcaps)
        if sub:
            run.font.subscript = True
        if sup:
            run.font.superscript = True
    return par


def _fmt(par, align=None, before=0, after=0, first=None, left=None, hanging=None, keep=False,
         line=None):
    pf = par.paragraph_format
    if align is not None:
        pf.alignment = align
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if first is not None:
        pf.first_line_indent = first
    if left is not None:
        pf.left_indent = left
    if hanging is not None:
        pf.left_indent = hanging
        pf.first_line_indent = -hanging
    if keep:
        pf.keep_with_next = True
    if line is not None:
        pf.line_spacing = Pt(line)
    return par


def _cols(section, num):
    sect = section._sectPr
    cols = sect.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(COL_GAP.twips)))
    cols.set(qn("w:equalWidth"), "1")


def _set_cell_border(cell, **edges):
    tcpr = cell._tc.get_or_add_tcPr()
    borders = tcpr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcpr.append(borders)
    for edge, sz in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")
        borders.append(el)


def _cell_margins(table, top=20, bottom=20, left=50, right=50):
    tblpr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(v))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    # schema order: tblCellMar comes before tblLook / tblCaption
    look = tblpr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)
    else:
        tblpr.append(mar)


def roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out, n = out + s, n - v
    return out


class IEEEDoc:
    """A paper being written top to bottom."""

    def __init__(self):
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = FONT, Pt(10)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        st.paragraph_format.space_before = st.paragraph_format.space_after = Pt(0)
        for name, size, italic in (("Heading 1", 10, False), ("Heading 2", 10, True),
                                   ("Heading 3", 10, True)):
            h = self.doc.styles[name]
            h.font.name, h.font.size, h.font.italic, h.font.bold = FONT, Pt(size), italic, False
            h.font.color.rgb = RGBColor(0, 0, 0)
            rpr = h.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.insert(0, fonts)
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                fonts.set(qn(attr), FONT)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if fonts.get(qn(attr)) is not None:
                    del fonts.attrib[qn(attr)]
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = PAGE_W, PAGE_H
        sec.top_margin, sec.bottom_margin = MARGIN_TOP, MARGIN_BOTTOM
        sec.left_margin = sec.right_margin = MARGIN_SIDE
        sec.header_distance, sec.footer_distance = Mm(12.7), Mm(12.7)
        _cols(sec, 1)
        self.ncols = 1
        self.table_no = 0
        self.fig_no = 0
        self.eq_no = 0

    # ---- sections ------------------------------------------------------------------------
    def columns(self, n):
        if n == self.ncols:
            return
        sec = self.doc.add_section(WD_SECTION.CONTINUOUS)
        _cols(sec, n)
        self.ncols = n

    # ---- title block ---------------------------------------------------------------------
    def title(self, text):
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.CENTER, after=10)
        add_runs(p, text, size=22)

    def authors(self, people):
        """``people``: list of (name, [affiliation lines], email)."""
        t = self.doc.add_table(rows=1, cols=len(people))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, (name, aff, email) in zip(t.rows[0].cells, people):
            cell.width = int(TEXT_W / len(people))
            p = cell.paragraphs[0]
            _fmt(p, WD_ALIGN_PARAGRAPH.CENTER)
            add_runs(p, name, size=11)
            for line in aff:
                q = _fmt(cell.add_paragraph(), WD_ALIGN_PARAGRAPH.CENTER)
                add_runs(q, line, size=9, italic=True)
            q = _fmt(cell.add_paragraph(), WD_ALIGN_PARAGRAPH.CENTER)
            add_runs(q, email, size=9)
        _fmt(self.doc.add_paragraph(), after=6)

    # ---- body ----------------------------------------------------------------------------
    def abstract(self, label, text):
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.JUSTIFY, after=4, first=Mm(3.6))
        add_runs(p, f"{label}—", size=9, bold=True, italic=True)
        add_runs(p, text, size=9, bold=True)

    def h1(self, text):
        p = self.doc.add_paragraph(style="Heading 1")
        _fmt(p, WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, keep=True)
        add_runs(p, text, size=10, smallcaps=True)

    def h2(self, text):
        p = self.doc.add_paragraph(style="Heading 2")
        _fmt(p, WD_ALIGN_PARAGRAPH.LEFT, before=5, after=2, keep=True)
        add_runs(p, text, size=10, italic=True)

    def p(self, text, runin=None, indent=True, size=10, style=None):
        p = self.doc.add_paragraph(style=style) if style else self.doc.add_paragraph()
        _fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, first=Mm(3.6) if indent else Mm(0))
        if runin:
            add_runs(p, runin + ": ", size=size, italic=True)
        add_runs(p, text, size=size)
        return p

    def item(self, label, text, size=10):
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.JUSTIFY, hanging=Mm(5))
        p.paragraph_format.tab_stops.add_tab_stop(Mm(5))
        add_runs(p, f"{label}\t", size=size)
        add_runs(p, text, size=size)
        return p

    def eq(self, text):
        self.eq_no += 1
        width = COL_W if self.ncols == 2 else TEXT_W
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.LEFT, before=3, after=3)
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
        add_runs(p, "\t")
        add_runs(p, text, italic=False)
        add_runs(p, f"\t({self.eq_no})")
        return self.eq_no

    def code(self, lines, size=7.5):
        """A verbatim block (prompts): Courier, no justification, left rule."""
        for line in lines:
            p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.LEFT, left=Mm(2))
            run = p.add_run(line if line else " ")
            run.font.name, run.font.size = "Courier New", Pt(size)
        _fmt(self.doc.add_paragraph(), after=2)

    # ---- floats --------------------------------------------------------------------------
    def table(self, caption, header, rows, widths=None, wide=False, size=8, align=None,
              note=None):
        """IEEE table: 'TABLE N' + caption above, top/bottom rules, header rule."""
        if wide:
            self.columns(1)
        self.table_no += 1
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.CENTER, before=6, keep=True)
        add_runs(p, f"Table {roman(self.table_no)}", size=8, smallcaps=True)
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.CENTER, after=3, keep=True)
        add_runs(p, caption, size=8, smallcaps=True)
        avail = TEXT_W if (wide or self.ncols == 1) else COL_W
        n = len(header)
        widths = widths or [1] * n
        tot = sum(widths)
        t = self.doc.add_table(rows=1 + len(rows), cols=n)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        _cell_margins(t)
        align = align or (["l"] + ["c"] * (n - 1))
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER,
                "r": WD_ALIGN_PARAGRAPH.RIGHT}
        for r, values in enumerate([header] + rows):
            for c, v in enumerate(values):
                cell = t.rows[r].cells[c]
                cell.width = int(avail * widths[c] / tot)
                par = cell.paragraphs[0]
                _fmt(par, amap[align[c]])
                add_runs(par, str(v), size=size, bold=False)
                edges = {}
                if r == 0:
                    edges["top"], edges["bottom"] = 8, 4
                if r == len(rows):
                    edges["bottom"] = 8
                if edges:
                    _set_cell_border(cell, **edges)
        if note:
            q = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.JUSTIFY, before=2)
            add_runs(q, note, size=7)
        _fmt(self.doc.add_paragraph(), after=4)
        if wide:
            self.columns(2)
        return self.table_no

    def figure(self, path, caption, wide=False, width_frac=1.0):
        if wide:
            self.columns(1)
        self.fig_no += 1
        avail = TEXT_W if (wide or self.ncols == 1) else COL_W
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.CENTER, before=6, keep=True)
        p.add_run().add_picture(str(path), width=int(avail * width_frac))
        q = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.JUSTIFY, before=2, after=8)
        add_runs(q, f"Fig. {self.fig_no}. ", size=8)
        add_runs(q, caption, size=8)
        if wide:
            self.columns(2)
        return self.fig_no

    def reference(self, n, text):
        p = _fmt(self.doc.add_paragraph(), WD_ALIGN_PARAGRAPH.JUSTIFY, hanging=Mm(6))
        p.paragraph_format.tab_stops.add_tab_stop(Mm(6))
        add_runs(p, f"[{n}]\t", size=8)
        add_runs(p, text, size=8)

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        zoom = self.doc.settings.element.find(qn("w:zoom"))
        if zoom is not None and zoom.get(qn("w:percent")) is None:
            zoom.set(qn("w:percent"), "100")         # required by the schema; Word may prompt without it
        props = self.doc.core_properties
        props.title = "TextGuard"
        props.author = "Aryan Manoj Waingankar, Devansh Kumar, Nivedita Mishra"
        self.doc.save(str(path))
