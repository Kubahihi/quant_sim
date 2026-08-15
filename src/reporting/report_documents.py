"""Production DOCX and PDF exports for final Evidence Studio workspaces.

The module has no mandatory dependency beyond QuantSim's existing runtime.
DOCX output is deterministic OOXML written with the standard library.  PDF
output prefers ReportLab when available and falls back to Matplotlib, which is
already a pinned production dependency.  Both public exporters return bytes so
the caller remains responsible for download, object storage, or file naming.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from io import BytesIO
import math
import re
import textwrap
from typing import Any
from xml.sax.saxutils import escape as xml_escape
import zipfile

from src.reporting.evidence_studio import build_export_ready_report


_DOCX_REQUIRED_PARTS = frozenset(
    {
        "[Content_Types].xml",
        "_rels/.rels",
        "docProps/app.xml",
        "docProps/core.xml",
        "word/document.xml",
        "word/styles.xml",
        "word/_rels/document.xml.rels",
        "word/header1.xml",
        "word/footer1.xml",
    }
)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _text(value: Any, default: str = "-") -> str:
    result = str(value or "").strip()
    return result or default


def _single_line(value: Any, default: str = "-") -> str:
    return " ".join(_text(value, default).split())


def _safe_xml(value: Any) -> str:
    return xml_escape(str(value or ""), {'"': "&quot;", "'": "&apos;"})


def _percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "-"
    return f"{number:.2%}" if math.isfinite(number) else "-"


def _money(value: Any, currency: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "-"
    if not math.isfinite(number):
        return "-"
    suffix = f" {currency}" if currency else ""
    return f"{number:,.2f}{suffix}"


def _display_date(value: Any) -> str:
    raw = _single_line(value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _split_paragraphs(value: Any) -> list[str]:
    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [" ".join(part.split()) for part in re.split(r"\n\s*\n", raw)]
    return [part for part in parts if part]


def _report_blocks(model: Mapping[str, Any], *, include_appendices: bool) -> list[dict[str, Any]]:
    """Create one renderer-neutral stream of semantic report blocks."""
    metadata = model["metadata"]
    snapshot = model.get("portfolio_snapshot") or {}
    attribution = model.get("performance_attribution") or {}
    blocks: list[dict[str, Any]] = [
        {
            "kind": "masthead",
            "title": _single_line(metadata.get("title"), "Investment report"),
            "subtitle": "Wharton Investment Competition - Evidence-backed team report",
            "rows": [
                ("Deliverable", _single_line(metadata.get("report_type")).replace("_", " ").title()),
                ("Report ID", _single_line(metadata.get("report_id"))),
                ("Schema", _single_line(metadata.get("schema_version"))),
                ("Finalised", _display_date(metadata.get("finalised_at"))),
                ("Approved by", _single_line(metadata.get("finalised_by"))),
            ],
        },
        {"kind": "heading", "level": 1, "text": "Portfolio snapshot"},
        {
            "kind": "definition",
            "label": "As of",
            "text": _display_date(snapshot.get("as_of")),
        },
        {
            "kind": "definition",
            "label": "Source",
            "text": _single_line(snapshot.get("source")),
        },
        {
            "kind": "definition",
            "label": "Reconciliation",
            "text": _single_line(snapshot.get("reconciliation_id")),
        },
    ]
    position_rows = []
    for position in snapshot.get("positions", []):
        position_rows.append(
            [
                _single_line(position.get("ticker") or position.get("security_id")),
                _percent(position.get("weight")),
                _money(position.get("market_value"), _single_line(position.get("currency"), "")),
            ]
        )
    if position_rows:
        blocks.append(
            {
                "kind": "table",
                "headers": ["Security", "Weight", "Market value"],
                "rows": position_rows,
                "widths": [0.34, 0.18, 0.48],
            }
        )

    if attribution:
        blocks.extend(
            [
                {"kind": "heading", "level": 1, "text": "Performance attribution"},
                {
                    "kind": "paragraph",
                    "text": (
                        f"As of {_display_date(attribution.get('as_of'))}, the portfolio returned "
                        f"{_percent(attribution.get('portfolio_return'))} versus "
                        f"{_percent(attribution.get('benchmark_return'))} for "
                        f"{_single_line(attribution.get('benchmark'))}. Active return was "
                        f"{_percent(attribution.get('active_return'))}."
                    ),
                },
                {
                    "kind": "definition",
                    "label": "Methodology",
                    "text": _single_line(attribution.get("methodology")),
                },
                {
                    "kind": "table",
                    "headers": ["Source", "Contribution"],
                    "rows": [
                        [_single_line(item.get("label")), _percent(item.get("contribution"))]
                        for item in attribution.get("contributions", [])
                    ]
                    + [["Unattributed residual", _percent(attribution.get("residual"))]],
                    "widths": [0.72, 0.28],
                },
            ]
        )

    blocks.append({"kind": "heading", "level": 1, "text": "Report narrative"})
    for section in model.get("sections", []):
        blocks.append(
            {
                "kind": "heading",
                "level": 2,
                "text": _single_line(section.get("title"), _single_line(section.get("id"))),
            }
        )
        blocks.append(
            {
                "kind": "definition",
                "label": "Owner / reviewer",
                "text": f"{_single_line(section.get('owner'))} / {_single_line(section.get('reviewer'))}",
            }
        )
        for paragraph in _split_paragraphs(section.get("content")):
            blocks.append({"kind": "paragraph", "text": paragraph})

        claims = section.get("claims", [])
        if claims:
            blocks.append({"kind": "heading", "level": 3, "text": "Evidence-backed claims"})
        for index, claim in enumerate(claims, start=1):
            blocks.append(
                {
                    "kind": "definition",
                    "label": f"Claim {index}",
                    "text": _single_line(claim.get("statement")),
                }
            )
            for evidence in claim.get("evidence", []):
                blocks.append(
                    {
                        "kind": "citation",
                        "text": (
                            f"Evidence: {_single_line(evidence.get('citation'))} - "
                            f"{_single_line(evidence.get('source_locator'))}"
                        ),
                    }
                )

        figures = section.get("figures", [])
        if figures:
            blocks.append({"kind": "heading", "level": 3, "text": "Figures"})
        for figure in figures:
            blocks.extend(
                [
                    {
                        "kind": "definition",
                        "label": _single_line(figure.get("title"), "Figure"),
                        "text": _single_line(figure.get("caption")),
                    },
                    {
                        "kind": "citation",
                        "text": (
                            f"Artifact: {_single_line(figure.get('artifact_locator'))}; "
                            f"data as of {_display_date(figure.get('data_as_of'))}."
                        ),
                    },
                ]
            )

        case_studies = section.get("case_studies", [])
        if case_studies:
            blocks.append({"kind": "heading", "level": 3, "text": "Decision case studies"})
        for case_study in case_studies:
            blocks.extend(
                [
                    {
                        "kind": "heading",
                        "level": 3,
                        "text": (
                            f"{_single_line(case_study.get('ticker'))}: "
                            f"{_single_line(case_study.get('title'))}"
                        ),
                    },
                    {
                        "kind": "definition",
                        "label": "Decision",
                        "text": _single_line(case_study.get("decision_id")),
                    },
                    {
                        "kind": "definition",
                        "label": "Process",
                        "text": _single_line(case_study.get("process_summary")),
                    },
                    {
                        "kind": "definition",
                        "label": "Outcome",
                        "text": _single_line(case_study.get("outcome_summary")),
                    },
                    {
                        "kind": "definition",
                        "label": "Lesson",
                        "text": _single_line(case_study.get("lesson")),
                    },
                ]
            )

    if include_appendices:
        blocks.extend(
            [
                {"kind": "page_break"},
                {"kind": "heading", "level": 1, "text": "Evidence and audit appendix"},
            ]
        )
        evidence_by_id: dict[str, dict[str, Any]] = {}
        for section in model.get("sections", []):
            for claim in section.get("claims", []):
                for evidence in claim.get("evidence", []):
                    evidence_by_id[evidence["evidence_id"]] = evidence
        for evidence_id in sorted(evidence_by_id):
            evidence = evidence_by_id[evidence_id]
            blocks.extend(
                [
                    {
                        "kind": "heading",
                        "level": 3,
                        "text": _single_line(evidence.get("title"), evidence_id),
                    },
                    {
                        "kind": "definition",
                        "label": "Citation",
                        "text": _single_line(evidence.get("citation")),
                    },
                    {
                        "kind": "definition",
                        "label": "Source",
                        "text": _single_line(evidence.get("source_locator")),
                    },
                    {
                        "kind": "definition",
                        "label": "Verified by",
                        "text": _single_line(evidence.get("verified_by")),
                    },
                ]
            )
        approvals = model.get("audit", {}).get("approvals", [])
        if approvals:
            blocks.extend(
                [
                    {"kind": "heading", "level": 2, "text": "Final approvals"},
                    {
                        "kind": "table",
                        "headers": ["Approver", "Decision", "Timestamp"],
                        "rows": [
                            [
                                _single_line(item.get("approver")),
                                _single_line(item.get("decision")).replace("_", " ").title(),
                                _display_date(item.get("decided_at")),
                            ]
                            for item in approvals
                        ],
                        "widths": [0.28, 0.27, 0.45],
                    },
                ]
            )
        blocks.append(
            {
                "kind": "citation",
                "text": f"Export integrity hash: {_single_line(model.get('export_hash'))}",
            }
        )
    return blocks


def _w_run(text: Any, *, bold: bool = False, italic: bool = False, color: str = "") -> str:
    properties = []
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    if color:
        properties.append(f'<w:color w:val="{_safe_xml(color)}"/>')
    rpr = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{_safe_xml(text)}</w:t></w:r>'


def _w_paragraph(
    text: Any = "",
    *,
    style: str = "Normal",
    runs: Sequence[tuple[Any, bool, bool, str]] | None = None,
    alignment: str = "",
    page_break: bool = False,
) -> str:
    ppr_parts = [f'<w:pStyle w:val="{_safe_xml(style)}"/>'] if style else []
    if alignment:
        ppr_parts.append(f'<w:jc w:val="{_safe_xml(alignment)}"/>')
    ppr = f"<w:pPr>{''.join(ppr_parts)}</w:pPr>" if ppr_parts else ""
    if page_break:
        body = '<w:r><w:br w:type="page"/></w:r>'
    elif runs is not None:
        body = "".join(
            _w_run(value, bold=bold, italic=italic, color=color)
            for value, bold, italic, color in runs
        )
    else:
        body = _w_run(text)
    return f"<w:p>{ppr}{body}</w:p>"


def _w_cell(text: Any, width: int, *, header: bool = False) -> str:
    fill = '<w:shd w:val="clear" w:color="auto" w:fill="F2F4F7"/>' if header else ""
    return (
        "<w:tc><w:tcPr>"
        f'<w:tcW w:w="{width}" w:type="dxa"/>{fill}'
        '<w:vAlign w:val="center"/>'
        "</w:tcPr>"
        + _w_paragraph(
            style="TableText",
            runs=[(_single_line(text), header, False, "0B2545" if header else "")],
        )
        + "</w:tc>"
    )


def _w_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], widths: Sequence[float]) -> str:
    dxa_widths = [int(round(9360 * value)) for value in widths]
    dxa_widths[-1] += 9360 - sum(dxa_widths)
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in dxa_widths)
    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="D7DBE2"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    table_rows = [
        "<w:tr><w:trPr><w:tblHeader/></w:trPr>"
        + "".join(_w_cell(value, width, header=True) for value, width in zip(headers, dxa_widths))
        + "</w:tr>"
    ]
    for row in rows:
        padded = list(row)[: len(headers)] + [""] * max(0, len(headers) - len(row))
        table_rows.append(
            "<w:tr>"
            + "".join(_w_cell(value, width) for value, width in zip(padded, dxa_widths))
            + "</w:tr>"
        )
    return (
        "<w:tbl><w:tblPr>"
        '<w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblInd w:w="120" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f"<w:tblBorders>{borders}</w:tblBorders>"
        '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:start w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(table_rows)}</w:tbl>"
        + _w_paragraph(style="TableSpacer")
    )


def _blocks_to_word_xml(blocks: Sequence[Mapping[str, Any]]) -> str:
    body: list[str] = []
    for block in blocks:
        kind = block["kind"]
        if kind == "masthead":
            body.extend(
                [
                    _w_paragraph(
                        block["title"], style="ReportTitle", alignment="left"
                    ),
                    _w_paragraph(block["subtitle"], style="ReportSubtitle"),
                    _w_table(
                        ["Report field", "Value"],
                        [[label, value] for label, value in block["rows"]],
                        [0.27, 0.73],
                    ),
                ]
            )
        elif kind == "heading":
            body.append(_w_paragraph(block["text"], style=f"Heading{block['level']}"))
        elif kind == "paragraph":
            body.append(_w_paragraph(block["text"], style="Normal"))
        elif kind == "definition":
            body.append(
                _w_paragraph(
                    style="Normal",
                    runs=[
                        (f"{block['label']}: ", True, False, "0B2545"),
                        (block["text"], False, False, ""),
                    ],
                )
            )
        elif kind == "citation":
            body.append(_w_paragraph(block["text"], style="Citation"))
        elif kind == "table":
            body.append(_w_table(block["headers"], block["rows"], block["widths"]))
        elif kind == "page_break":
            body.append(_w_paragraph(page_break=True))
    return "".join(body)


def _docx_styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{_W_NS}">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:color w:val="222222"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="0" w:after="120" w:line="264" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ReportTitle"><w:name w:val="Report Title"/><w:basedOn w:val="Normal"/><w:next w:val="ReportSubtitle"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="0" w:after="80"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="0B2545"/><w:sz w:val="48"/><w:szCs w:val="48"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ReportSubtitle"><w:name w:val="Report Subtitle"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="0" w:after="240"/></w:pPr><w:rPr><w:i/><w:color w:val="58697A"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="320" w:after="160"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="160" w:after="80"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:color w:val="1F4D78"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Citation"><w:name w:val="Citation"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="80" w:after="80"/><w:ind w:left="240"/></w:pPr><w:rPr><w:i/><w:color w:val="58697A"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableSpacer"><w:name w:val="Table Spacer"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="80"/><w:sz w:val="4"/></w:pPr></w:style>
</w:styles>'''


def _docx_parts(model: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    metadata = model["metadata"]
    report_title = _single_line(metadata.get("title"), "Investment report")
    report_id = _single_line(metadata.get("report_id"))
    finalised_at = _single_line(metadata.get("finalised_at"), "2000-01-01T00:00:00+00:00")
    try:
        core_date = datetime.fromisoformat(finalised_at.replace("Z", "+00:00")).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except ValueError:
        core_date = "2000-01-01T00:00:00Z"
    document_body = _blocks_to_word_xml(blocks)
    return {
        "[Content_Types].xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''',
        "word/_rels/document.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
</Relationships>''',
        "word/styles.xml": _docx_styles_xml(),
        "word/document.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{_W_NS}" xmlns:r="{_R_NS}"><w:body>{document_body}<w:sectPr>
  <w:headerReference w:type="default" r:id="rId2"/><w:footerReference w:type="default" r:id="rId3"/>
  <w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
  <w:cols w:space="708"/><w:docGrid w:linePitch="360"/>
</w:sectPr></w:body></w:document>''',
        "word/header1.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="{_W_NS}"><w:p><w:pPr><w:spacing w:after="0"/><w:pBdr><w:bottom w:val="single" w:sz="4" w:space="4" w:color="D7DBE2"/></w:pBdr></w:pPr>{_w_run(report_title, bold=True, color="58697A")}</w:p></w:hdr>''',
        "word/footer1.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{_W_NS}"><w:p><w:pPr><w:jc w:val="right"/><w:spacing w:after="0"/></w:pPr>{_w_run(report_id + "  |  Page ", color="58697A")}<w:fldSimple w:instr=" PAGE ">{_w_run("1", color="58697A")}</w:fldSimple></w:p></w:ftr>''',
        "docProps/core.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{_safe_xml(report_title)}</dc:title><dc:subject>Evidence-backed investment report</dc:subject><dc:creator>QuantSim Evidence Studio</dc:creator><cp:lastModifiedBy>{_safe_xml(metadata.get("finalised_by"))}</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{core_date}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{core_date}</dcterms:modified><cp:keywords>investment; evidence; audit; Wharton</cp:keywords>
</cp:coreProperties>''',
        "docProps/app.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>QuantSim</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>1.0</AppVersion></Properties>''',
    }


def inspect_docx_bytes(data: bytes) -> dict[str, Any]:
    """Structurally validate DOCX bytes and return package diagnostics."""
    if not isinstance(data, bytes) or not data.startswith(b"PK"):
        raise ValueError("DOCX data must be a ZIP-based Office document.")
    try:
        with zipfile.ZipFile(BytesIO(data)) as package:
            names = set(package.namelist())
            missing = sorted(_DOCX_REQUIRED_PARTS - names)
            corrupt_part = package.testzip()
            document_xml = package.read("word/document.xml") if not missing else b""
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("DOCX package is invalid.") from exc
    if missing:
        raise ValueError("DOCX package is missing required parts: " + ", ".join(missing))
    if corrupt_part:
        raise ValueError(f"DOCX package contains a corrupt part: {corrupt_part}.")
    try:
        from xml.etree import ElementTree

        root = ElementTree.fromstring(document_xml)
        visible_text = " ".join(
            node.text or "" for node in root.iter(f"{{{_W_NS}}}t") if node.text
        )
        paragraph_count = sum(1 for _ in root.iter(f"{{{_W_NS}}}p"))
        table_count = sum(1 for _ in root.iter(f"{{{_W_NS}}}tbl"))
    except ElementTree.ParseError as exc:
        raise ValueError("DOCX main document XML is invalid.") from exc
    return {
        "is_valid": True,
        "size_bytes": len(data),
        "part_count": len(names),
        "paragraph_count": paragraph_count,
        "table_count": table_count,
        "text": visible_text,
    }


def generate_evidence_report_docx(
    workspace: Mapping[str, Any],
    *,
    include_appendices: bool = True,
) -> bytes:
    """Convert a final Evidence Studio workspace into a production DOCX."""
    model = build_export_ready_report(workspace)
    blocks = _report_blocks(model, include_appendices=bool(include_appendices))
    parts = _docx_parts(model, blocks)
    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in sorted(parts):
            package.writestr(name, parts[name].encode("utf-8"))
    data = output.getvalue()
    inspect_docx_bytes(data)
    return data


def _reportlab_pdf(model: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_regular = "Helvetica"
    font_bold = "Helvetica-Bold"
    try:
        from matplotlib import font_manager

        regular_path = font_manager.findfont("DejaVu Sans")
        bold_path = font_manager.findfont("DejaVu Sans", fontext="ttf")
        pdfmetrics.registerFont(TTFont("QuantSimSans", regular_path))
        # Matplotlib may resolve the same universal TTF; registering it under a
        # bold name still preserves Unicode even when its weight is regular.
        pdfmetrics.registerFont(TTFont("QuantSimSansBold", bold_path))
        font_regular = "QuantSimSans"
        font_bold = "QuantSimSansBold"
    except Exception:
        pass

    output = BytesIO()
    metadata = model["metadata"]
    title = _single_line(metadata.get("title"), "Investment report")
    report_id = _single_line(metadata.get("report_id"))
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=0.82 * inch,
        bottomMargin=0.78 * inch,
        title=title,
        author="QuantSim Evidence Studio",
        subject="Evidence-backed investment report",
    )
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "QuantSimBody",
        parent=styles["BodyText"],
        fontName=font_regular,
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    title_style = ParagraphStyle(
        "QuantSimTitle",
        parent=body,
        fontName=font_bold,
        fontSize=23,
        leading=27,
        textColor=colors.HexColor("#0B2545"),
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "QuantSimSubtitle",
        parent=body,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#58697A"),
        spaceAfter=14,
    )
    headings = {
        1: ParagraphStyle(
            "QuantSimH1",
            parent=body,
            fontName=font_bold,
            fontSize=16,
            leading=19,
            textColor=colors.HexColor("#2E74B5"),
            spaceBefore=14,
            spaceAfter=7,
            keepWithNext=True,
        ),
        2: ParagraphStyle(
            "QuantSimH2",
            parent=body,
            fontName=font_bold,
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2E74B5"),
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        3: ParagraphStyle(
            "QuantSimH3",
            parent=body,
            fontName=font_bold,
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1F4D78"),
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
    }
    citation = ParagraphStyle(
        "QuantSimCitation",
        parent=body,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#58697A"),
        leftIndent=12,
        spaceBefore=3,
        spaceAfter=4,
    )

    def paragraph(value: Any, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(_safe_xml(value).replace("\n", "<br/>"), style)

    story: list[Any] = []
    for block in blocks:
        kind = block["kind"]
        if kind == "masthead":
            story.extend(
                [
                    paragraph(block["title"], title_style),
                    paragraph(block["subtitle"], subtitle),
                ]
            )
            rows = [[paragraph("Report field", body), paragraph("Value", body)]] + [
                [paragraph(label, body), paragraph(value, body)] for label, value in block["rows"]
            ]
            table = Table(rows, colWidths=[1.75 * inch, 4.75 * inch], repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), font_regular),
                        ("FONTNAME", (0, 0), (-1, 0), font_bold),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DBE2")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 5)])
        elif kind == "heading":
            story.append(paragraph(block["text"], headings[int(block["level"])]))
        elif kind == "paragraph":
            story.append(paragraph(block["text"]))
        elif kind == "definition":
            story.append(
                Paragraph(
                    f'<font name="{font_bold}" color="#0B2545">{_safe_xml(block["label"])}:</font> '
                    f'{_safe_xml(block["text"])}',
                    body,
                )
            )
        elif kind == "citation":
            story.append(paragraph(block["text"], citation))
        elif kind == "table":
            widths = [6.5 * inch * float(value) for value in block["widths"]]
            rows = [[paragraph(value, body) for value in block["headers"]]] + [
                [paragraph(value, body) for value in row] for row in block["rows"]
            ]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), font_regular),
                        ("FONTNAME", (0, 0), (-1, 0), font_bold),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DBE2")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 5)])
        elif kind == "page_break":
            story.append(PageBreak())

    def page_chrome(canvas: Any, _document: Any) -> None:
        canvas.saveState()
        canvas.setTitle(title)
        canvas.setAuthor("QuantSim Evidence Studio")
        canvas.setFont(font_regular, 8)
        canvas.setFillColor(colors.HexColor("#58697A"))
        canvas.drawString(inch, 10.55 * inch, title[:90])
        canvas.setStrokeColor(colors.HexColor("#D7DBE2"))
        canvas.setLineWidth(0.4)
        canvas.line(inch, 10.43 * inch, 7.5 * inch, 10.43 * inch)
        canvas.drawRightString(7.5 * inch, 0.48 * inch, f"{report_id}  |  Page {canvas.getPageNumber()}")
        canvas.restoreState()

    # A leading pair should stay together while tables remain splittable.
    if len(story) >= 2:
        story[:2] = [KeepTogether(story[:2])]
    document.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return output.getvalue()


class _MatplotlibPaginator:
    """Small layout engine used only when ReportLab is unavailable."""

    def __init__(self, pdf: Any, *, title: str, report_id: str) -> None:
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D

        self._Figure = Figure
        self._Line2D = Line2D
        self._pdf = pdf
        self._title = title
        self._report_id = report_id
        self._page_number = 0
        self._figure: Any = None
        self._y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        if self._figure is not None:
            self._finish_page()
        self._page_number += 1
        self._figure = self._Figure(figsize=(8.5, 11), facecolor="white")
        self._figure.text(0.105, 0.965, self._title[:100], fontsize=8, color="#58697A")
        self._figure.add_artist(
            self._Line2D(
                [0.105, 0.895],
                [0.952, 0.952],
                color="#D7DBE2",
                linewidth=0.6,
            )
        )
        self._figure.text(
            0.895,
            0.032,
            f"{self._report_id}  |  Page {self._page_number}",
            fontsize=8,
            color="#58697A",
            ha="right",
        )
        self._y = 0.918

    def _finish_page(self) -> None:
        self._pdf.savefig(self._figure, bbox_inches=None)
        self._figure.clear()

    def finish(self) -> None:
        if self._figure is not None:
            self._finish_page()
            self._figure = None

    def _ensure(self, height: float) -> None:
        if self._y - height < 0.075:
            self._new_page()

    @staticmethod
    def _wrap(value: Any, width: int) -> list[str]:
        text = _single_line(value)
        return textwrap.wrap(
            text,
            width=max(8, width),
            break_long_words=True,
            break_on_hyphens=True,
        ) or [""]

    def text(
        self,
        value: Any,
        *,
        size: float = 10.5,
        color: str = "#222222",
        weight: str = "normal",
        italic: bool = False,
        before: float = 0.0,
        after: float = 0.009,
        indent: float = 0.0,
        width_chars: int | None = None,
    ) -> None:
        width = width_chars or max(20, int((93 - 100 * indent) * 10.5 / size))
        lines = self._wrap(value, width)
        line_height = 0.0175 * size / 10.5
        height = before + len(lines) * line_height + after
        self._ensure(height)
        self._y -= before
        for line in lines:
            self._figure.text(
                0.105 + indent,
                self._y,
                line,
                fontsize=size,
                color=color,
                fontweight=weight,
                fontstyle="italic" if italic else "normal",
                va="top",
            )
            self._y -= line_height
        self._y -= after

    def definition(self, label: Any, value: Any) -> None:
        self.text(f"{_single_line(label)}: {_single_line(value)}", size=10.5, after=0.006)

    def heading(self, value: Any, level: int) -> None:
        tokens = {
            1: (16, "#2E74B5", 0.016, 0.012),
            2: (13, "#2E74B5", 0.013, 0.009),
            3: (11.5, "#1F4D78", 0.010, 0.007),
        }
        size, color, before, after = tokens[level]
        self.text(value, size=size, color=color, weight="bold", before=before, after=after)

    def table(
        self,
        headers: Sequence[Any],
        rows: Sequence[Sequence[Any]],
        widths: Sequence[float],
    ) -> None:
        from matplotlib.patches import Rectangle

        x0 = 0.105
        total_width = 0.79
        column_widths = [total_width * float(value) for value in widths]

        def draw_row(values: Sequence[Any], *, header: bool) -> None:
            wrapped: list[list[str]] = []
            for value, fraction in zip(values, widths):
                wrapped.append(self._wrap(value, max(8, int(82 * float(fraction)))))
            row_height = max(len(lines) for lines in wrapped) * 0.015 + 0.012
            self._ensure(row_height)
            x = x0
            for lines, column_width in zip(wrapped, column_widths):
                rectangle = Rectangle(
                    (x, self._y - row_height),
                    column_width,
                    row_height,
                    transform=self._figure.transFigure,
                    facecolor="#F2F4F7" if header else "white",
                    edgecolor="#D7DBE2",
                    linewidth=0.55,
                )
                self._figure.add_artist(rectangle)
                for index, line in enumerate(lines):
                    self._figure.text(
                        x + 0.007,
                        self._y - 0.007 - index * 0.015,
                        line,
                        fontsize=8.5,
                        color="#0B2545" if header else "#222222",
                        fontweight="bold" if header else "normal",
                        va="top",
                    )
                x += column_width
            self._y -= row_height

        self._ensure(0.05)
        draw_row(headers, header=True)
        for row in rows:
            padded = list(row)[: len(headers)] + [""] * max(0, len(headers) - len(row))
            # Repeat the table header if a row forces a new page.
            estimated = max(
                len(self._wrap(value, max(8, int(82 * float(fraction)))))
                for value, fraction in zip(padded, widths)
            ) * 0.015 + 0.012
            if self._y - estimated < 0.075:
                self._new_page()
                draw_row(headers, header=True)
            draw_row(padded, header=False)
        self._y -= 0.012


def _matplotlib_pdf(model: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]]) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import rc_context

    metadata = model["metadata"]
    title = _single_line(metadata.get("title"), "Investment report")
    report_id = _single_line(metadata.get("report_id"))
    output = BytesIO()
    with rc_context({"font.family": "DejaVu Sans", "pdf.compression": 0}):
        with PdfPages(
            output,
            metadata={
                "Title": title,
                "Author": "QuantSim Evidence Studio",
                "Subject": "Evidence-backed investment report",
                "Keywords": "investment evidence audit Wharton",
            },
        ) as pdf:
            paginator = _MatplotlibPaginator(pdf, title=title, report_id=report_id)
            for block in blocks:
                kind = block["kind"]
                if kind == "masthead":
                    paginator.text(
                        block["title"],
                        size=23,
                        color="#0B2545",
                        weight="bold",
                        after=0.006,
                    )
                    paginator.text(
                        block["subtitle"],
                        size=12,
                        color="#58697A",
                        italic=True,
                        after=0.015,
                    )
                    paginator.table(
                        ["Report field", "Value"],
                        [[label, value] for label, value in block["rows"]],
                        [0.27, 0.73],
                    )
                elif kind == "heading":
                    paginator.heading(block["text"], int(block["level"]))
                elif kind == "paragraph":
                    paginator.text(block["text"])
                elif kind == "definition":
                    paginator.definition(block["label"], block["text"])
                elif kind == "citation":
                    paginator.text(
                        block["text"],
                        size=9,
                        color="#58697A",
                        italic=True,
                        indent=0.018,
                        after=0.006,
                    )
                elif kind == "table":
                    paginator.table(block["headers"], block["rows"], block["widths"])
                elif kind == "page_break":
                    paginator._new_page()
            paginator.finish()
    return output.getvalue()


def inspect_pdf_bytes(data: bytes) -> dict[str, Any]:
    """Perform dependency-free structural checks on generated PDF bytes."""
    if not isinstance(data, bytes) or not data.startswith(b"%PDF-"):
        raise ValueError("PDF data must start with a valid PDF signature.")
    if b"%%EOF" not in data[-1024:]:
        raise ValueError("PDF data is missing its end-of-file marker.")
    page_count = len(re.findall(rb"/Type\s*/Page(?!s)\b", data))
    if page_count <= 0:
        raise ValueError("PDF contains no page objects.")
    return {
        "is_valid": True,
        "size_bytes": len(data),
        "page_count": page_count,
        "pdf_version": data[5:8].decode("ascii", errors="replace"),
    }


def generate_evidence_report_pdf(
    workspace: Mapping[str, Any],
    *,
    include_appendices: bool = True,
) -> bytes:
    """Convert a final Evidence Studio workspace into a production PDF."""
    model = build_export_ready_report(workspace)
    blocks = _report_blocks(model, include_appendices=bool(include_appendices))
    try:
        data = _reportlab_pdf(model, blocks)
    except ImportError:
        data = _matplotlib_pdf(model, blocks)
    inspect_pdf_bytes(data)
    return data


def generate_evidence_report_documents(
    workspace: Mapping[str, Any],
    *,
    include_appendices: bool = True,
) -> dict[str, bytes]:
    """Generate both download-ready document formats from one frozen model."""
    return {
        "docx": generate_evidence_report_docx(
            workspace, include_appendices=include_appendices
        ),
        "pdf": generate_evidence_report_pdf(
            workspace, include_appendices=include_appendices
        ),
    }


__all__ = [
    "generate_evidence_report_documents",
    "generate_evidence_report_docx",
    "generate_evidence_report_pdf",
    "inspect_docx_bytes",
    "inspect_pdf_bytes",
]
