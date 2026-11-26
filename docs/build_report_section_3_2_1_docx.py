from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "report_section_3_2_1_fk.md"
DOCX_PATH = ROOT / "docs" / "report_section_3_2_1_fk.docx"


def formula_to_word_text(formula: str) -> str:
    compact = re.sub(r"\s+", "", formula)
    if compact.startswith(r"\mathbfz^\ast=\arg\min"):
        return "𝐳* = arg min_𝐳 1/2 ‖𝐄(𝐳; 𝐈, {𝐦ᵢᵇ})‖₂²,"
    if compact.startswith(r"\mathbfx_i(s)="):
        return "𝐱ᵢ(s) = [𝐩ᵢ(s), 𝐪ᵢ(s), 𝐧ᵢ(s), 𝛕ᵢ(s)],"
    if r"\mathbfv_i-\mathbfv_i^\ast" in compact and r"\mathbfu_i-\mathbfu_i^\ast" in compact:
        return "𝐯ᵢ − 𝐯ᵢ* = 𝐊ₛₑ,ᵢ⁻¹ 𝐑ᵢᵀ𝐧ᵢ,    𝐮ᵢ − 𝐮ᵢ* = 𝐊ᵦₜ,ᵢ⁻¹ 𝐑ᵢᵀ𝛕ᵢ,"
    if compact.startswith(r"\mathbfm_i=\mathbfR_i"):
        return "𝐦ᵢ = 𝐑ᵢ𝐦ᵢᵇ."
    if r"\boldsymbol{\tau}_{m,i}" in compact and r"\mathbff_{m,i}" in compact:
        return "𝛕ₘ,ᵢ = 𝐦ᵢ × 𝐁(𝐩ᵢ, 𝐈),    𝐟ₘ,ᵢ = ∇(𝐦ᵢᵀ𝐁(𝐩ᵢ, 𝐈))."
    if compact.startswith(r"\mathbfE=\left["):
        return "𝐄 = [𝐄_flex, 𝐄_rigid, 𝐄_joint, 𝐄_bc, 𝐄_tip]ᵀ,"
    if compact.startswith(r"\mathbfJ_k="):
        return "𝐉ₖ = ∂𝐄/∂𝐳 |_(𝐳=𝐳ₖ)."
    if r"\mathbfJ_k^\mathrmT\mathbfJ_k+\lambda_k\mathbfI" in compact:
        return "(𝐉ₖᵀ𝐉ₖ + λₖ𝐈) Δ𝐳ₖ = −𝐉ₖᵀ𝐄ₖ."
    return formula.strip()


INLINE_REPL = [
    (r"\mathbf z^\ast", "𝐳*"),
    (r"\mathbf z_k", "𝐳ₖ"),
    (r"\Delta\mathbf z_k", "Δ𝐳ₖ"),
    (r"\mathbf E_k=\mathbf E(\mathbf z_k)", "𝐄ₖ=𝐄(𝐳ₖ)"),
    (r"\mathbf E_k=\mathbf E(𝐳_k)", "𝐄ₖ=𝐄(𝐳ₖ)"),
    (r"\{\mathbf m_i^b\}", "{𝐦ᵢᵇ}"),
    (r"\mathbf m_i^b", "𝐦ᵢᵇ"),
    (r"\mathbf m_i", "𝐦ᵢ"),
    (r"\mathbf p_i(s)", "𝐩ᵢ(s)"),
    (r"\mathbf q_i(s)", "𝐪ᵢ(s)"),
    (r"\mathbf n_i(s)", "𝐧ᵢ(s)"),
    (r"\boldsymbol{\tau}_i(s)", "𝛕ᵢ(s)"),
    (r"\mathbf v_i^\ast", "𝐯ᵢ*"),
    (r"\mathbf u_i^\ast", "𝐮ᵢ*"),
    (r"\mathbf v_i", "𝐯ᵢ"),
    (r"\mathbf u_i", "𝐮ᵢ"),
    (r"\mathbf K_{se,i}", "𝐊ₛₑ,ᵢ"),
    (r"\mathbf K_{bt,i}", "𝐊ᵦₜ,ᵢ"),
    (r"\mathbf p_i", "𝐩ᵢ"),
    (r"\mathbf I", "𝐈"),
    (r"\mathbf p_0", "𝐩₀"),
    (r"\mathbf Q_0", "𝐐₀"),
    (r"\mathbf z", "𝐳"),
    (r"\mathbf E(\cdot)", "𝐄(·)"),
    (r"\mathbf E_{\mathrm{flex}}", "𝐄_flex"),
    (r"\mathbf E_{\mathrm{rigid}}", "𝐄_rigid"),
    (r"\mathbf E_{\mathrm{joint}}", "𝐄_joint"),
    (r"\mathbf E_{\mathrm{bc}}", "𝐄_bc"),
    (r"\mathbf E_{\mathrm{tip}}", "𝐄_tip"),
    (r"\mathbf E", "𝐄"),
    (r"\mathbf z_k+\Delta𝐳_k", "𝐳ₖ+Δ𝐳ₖ"),
    (r"𝐳_k+\Delta𝐳_k", "𝐳ₖ+Δ𝐳ₖ"),
    (r"𝐳_k+Δ𝐳ₖ", "𝐳ₖ+Δ𝐳ₖ"),
    (r"\mathbf B", "𝐁"),
    (r"\(xOz\)", "xOz"),
]


def clean_inline_math(text: str) -> str:
    for src, dst in INLINE_REPL:
        text = text.replace(src, dst)
    text = text.replace("^\\ast", "*").replace("^\\mathrm T", "ᵀ").replace("^\\mathrmT", "ᵀ")
    text = re.sub(r"\\\((.*?)\\\)", lambda m: m.group(1), text)
    for src, dst in INLINE_REPL:
        text = text.replace(src, dst)
    text = re.sub(r"\\[a-zA-Z]+(?:\{([^{}]+)\})?", lambda m: m.group(1) if m.group(1) else "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def w_text(text: str, *, bold: bool = False, size: int = 24, font: str = "SimSun") -> str:
    b = "<w:b/>" if bold else ""
    escaped = html.escape(text)
    return (
        "<w:r><w:rPr>"
        f"<w:rFonts w:ascii=\"Times New Roman\" w:eastAsia=\"{font}\" w:hAnsi=\"Times New Roman\"/>"
        f"{b}<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        "</w:rPr>"
        f"<w:t xml:space=\"preserve\">{escaped}</w:t></w:r>"
    )


def paragraph(text: str, *, style: str | None = None, bold: bool = False, size: int = 24, indent: bool = True) -> str:
    ppr = ""
    if style:
        ppr += f"<w:pStyle w:val=\"{style}\"/>"
    if indent and not style:
        ppr += "<w:ind w:firstLine=\"480\"/>"
    ppr += "<w:spacing w:before=\"80\" w:after=\"80\" w:line=\"360\" w:lineRule=\"auto\"/>"
    return f"<w:p><w:pPr>{ppr}</w:pPr>{w_text(clean_inline_math(text), bold=bold, size=size)}</w:p>"


def omath_paragraph(formula: str) -> str:
    text = formula_to_word_text(formula)
    escaped = html.escape(text)
    return (
        "<w:p><w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"120\" w:after=\"120\"/></w:pPr>"
        "<m:oMathPara><m:oMath>"
        "<m:r><m:rPr><m:sty m:val=\"p\"/></m:rPr>"
        f"<m:t>{escaped}</m:t>"
        "</m:r></m:oMath></m:oMathPara></w:p>"
    )


def code_block(text: str) -> str:
    lines = text.rstrip().splitlines()
    body = []
    for line in lines:
        body.append(
            "<w:p><w:pPr><w:pStyle w:val=\"CodeBlock\"/><w:spacing w:after=\"0\"/>"
            "</w:pPr>"
            f"{w_text(line, size=20, font='SimSun')}"
            "</w:p>"
        )
    return "".join(body)


def parse_markdown(md: str) -> list[str]:
    blocks: list[str] = []
    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if para:
            blocks.append(paragraph("".join(para)))
            para = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            flush_para()
            blocks.append(paragraph(line[2:].strip(), style="Heading1", bold=True, size=32, indent=False))
            i += 1
            continue
        if line.strip() == r"\[":
            flush_para()
            formula_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                formula_lines.append(lines[i])
                i += 1
            blocks.append(omath_paragraph("\n".join(formula_lines)))
            i += 1
            continue
        if line.strip() == "```":
            flush_para()
            code_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                code_lines.append(lines[i])
                i += 1
            blocks.append(code_block("\n".join(code_lines)))
            i += 1
            continue
        if not line.strip():
            flush_para()
            i += 1
            continue
        para.append(line.strip())
        i += 1
    flush_para()
    return blocks


def make_docx_xml(body: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<w:body>
{body}
<w:sectPr>
<w:pgSz w:w="11906" w:h="16838"/>
<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
</w:sectPr>
</w:body>
</w:document>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal">
<w:name w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:after="80" w:line="360" w:lineRule="auto"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="SimSun" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading1">
<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="240" w:after="240"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="SimSun" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="CodeBlock">
<w:name w:val="CodeBlock"/><w:basedOn w:val="Normal"/>
<w:pPr><w:ind w:left="360"/><w:spacing w:before="0" w:after="0"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Consolas" w:eastAsia="SimSun" w:hAnsi="Consolas"/><w:sz w:val="20"/></w:rPr>
</w:style>
</w:styles>'''


def write_docx(document_xml: str) -> None:
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml())


def main() -> None:
    md = MD_PATH.read_text(encoding="utf-8")
    body = "\n".join(parse_markdown(md))
    write_docx(make_docx_xml(body))
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
