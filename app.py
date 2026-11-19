"""Small HTTP API for Header Mapper.

The page in public/ is a single form. This module does not store uploads:
each request reads the file into memory, calls mapper, and returns the result.
Nothing here writes sheet contents to disk or to a log.

Vercel serves public/ from the CDN and this module as one FastAPI function.
Request bodies on Vercel are limited to 4.5 MB, so uploads are rejected at 4 MB.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

import mapper

# Stay under Vercel's 4.5 MB request-body limit, including multipart overhead.
MAX_BODY_BYTES = 4 * 1024 * 1024
MODEL_RE = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")
PUBLIC = Path(__file__).resolve().parent / "public"

app = FastAPI(title="Header Mapper", docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(mapper.MapperError)
async def _mapper_error(_request, exc: mapper.MapperError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=400)


def _cell(value: object) -> str | int | float | bool:
    """JSON-safe cell. Missing values become an empty string, not the word None."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return str(value)
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _table(frame: pd.DataFrame, limit: int = 10) -> dict:
    columns = [str(column) for column in frame.columns]
    rows: list[list] = []
    for record in frame.head(limit).itertuples(index=False, name=None):
        rows.append([_cell(value) for value in record])
    return {"columns": columns, "rows": rows}


def _column_choices(table: mapper.LoadedTable) -> list[dict]:
    choices = []
    for column in table.frame.columns:
        name = str(column)
        original = table.originals.get(name, name)
        label = name if original == name else f"{name} (original: {original})"
        choices.append({"name": name, "label": label})
    return choices


async def _read_upload(upload: UploadFile) -> tuple[bytes, str]:
    data = await upload.read()
    if len(data) > MAX_BODY_BYTES:
        raise mapper.FileTooLargeError("This file is over the 4 MB upload limit.")
    filename = Path(upload.filename or "upload.csv").name
    return data, filename


def _load(data: bytes, filename: str, sheet: str, *, header_only: bool) -> mapper.LoadedTable:
    chosen = sheet.strip() or None
    return mapper.load_table(
        data,
        filename,
        sheet=chosen,
        header_only=header_only,
        max_bytes=MAX_BODY_BYTES,
    )


def _sheet_names(data: bytes, filename: str) -> list[str]:
    try:
        return mapper.list_sheets(data, filename)
    except mapper.MapperError:
        raise
    except Exception as exc:
        raise mapper.MapperError("The workbook could not be opened. Save it as .xlsx and retry.") from exc


def _mapping_payload(row: mapper.Mapping) -> dict:
    return {
        "target": row.target,
        "source": row.source,
        "confidence": row.confidence,
        "reason": row.reason,
        "status": row.status,
        "origin": row.origin,
    }


def _mappings_from_client(
    raw: str,
    source: mapper.LoadedTable,
    target: mapper.LoadedTable,
) -> list[mapper.Mapping]:
    try:
        items = json.loads(raw) if raw else []
    except json.JSONDecodeError as exc:
        raise mapper.MapperError("The mapping list was not valid. Match the headers again.") from exc
    if not isinstance(items, list):
        raise mapper.MapperError("The mapping list was not valid. Match the headers again.")

    by_target: dict[str, dict] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("target"), str):
            by_target[item["target"]] = item

    known_sources = {str(column) for column in source.frame.columns}
    used: set[str] = set()
    rows: list[mapper.Mapping] = []
    for column in target.frame.columns:
        target_name = str(column)
        item = by_target.get(target_name, {})
        source_name = item.get("source")
        if source_name is not None and not isinstance(source_name, str):
            raise mapper.MapperError("A mapped source column was not a text name.")
        if isinstance(source_name, str):
            source_name = source_name.strip()
        if not source_name or source_name == mapper.CLEAR:
            source_name = None
        if source_name and source_name not in known_sources:
            raise mapper.MapperError(f"Unknown source column '{source_name}'.")
        if source_name and source_name in used:
            raise mapper.MapperError(
                f"'{source_name}' is mapped more than once. Clear the extra row, then retry."
            )
        if source_name:
            used.add(source_name)
        try:
            confidence = float(item.get("confidence", 1.0 if source_name else 0.0))
        except (TypeError, ValueError):
            confidence = 1.0 if source_name else 0.0
        status = item.get("status")
        if status not in {"Matched", "Needs review", "Unmapped"}:
            status = "Matched" if source_name else "Unmapped"
        reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
        origin = item.get("origin") if isinstance(item.get("origin"), str) else "manual"
        rows.append(
            mapper.Mapping(target_name, source_name, confidence, reason, status, origin)
        )
    return rows


def _profiles(source: mapper.LoadedTable, target: mapper.LoadedTable) -> dict:
    """Masked column profiles for every column. The page filters these before display."""

    source_profiles = mapper.build_llm_payload(
        [mapper.profile_column(str(column), source.frame[column]) for column in source.frame.columns],
        [],
        headers_only=False,
    )["sources"]
    target_profiles = mapper.build_llm_payload(
        [],
        mapper.profile_headers([str(column) for column in target.frame.columns]),
        headers_only=False,
    )["targets"]
    return {"source_profiles": source_profiles, "target_profiles": target_profiles}


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _threshold(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise mapper.MapperError("The fuzzy threshold must be a number from 0.60 to 1.00.") from exc
    if number < 0.60 or number > 1:
        raise mapper.MapperError("The fuzzy threshold must be a number from 0.60 to 1.00.")
    return number


@app.post("/api/source")
async def inspect_source(file: UploadFile = File(...), sheet: str = Form("")) -> dict:
    data, filename = await _read_upload(file)
    table = _load(data, filename, sheet, header_only=False)
    preview = _table(table.frame)
    return {
        "filename": filename,
        "sheets": _sheet_names(data, filename),
        "sheet": table.sheet_name,
        "rows": table.row_count,
        "columns": int(table.frame.shape[1]),
        "choices": _column_choices(table),
        "duplicate_headers": table.duplicate_names,
        "identical_columns": mapper.find_identical_columns(table.frame),
        "duplicate_rows": mapper.duplicate_row_count(table.frame),
        "preview": preview,
    }


@app.post("/api/target")
async def inspect_target(file: UploadFile = File(...), sheet: str = Form("")) -> dict:
    data, filename = await _read_upload(file)
    table = _load(data, filename, sheet, header_only=True)
    if table.row_count != 0 or len(table.frame):
        raise mapper.MapperError("The target sheet was read with data rows. Retry the upload.")
    headers = [table.originals.get(str(column), str(column)) for column in table.frame.columns]
    return {
        "filename": filename,
        "sheets": _sheet_names(data, filename),
        "sheet": table.sheet_name,
        "headers": headers,
        "duplicate_headers": table.duplicate_names,
        "column_count": len(headers),
    }


@app.post("/api/match")
async def match_headers(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    source_sheet: str = Form(""),
    target_sheet: str = Form(""),
    threshold: float = Form(mapper.DEFAULT_FUZZY_THRESHOLD),
) -> dict:
    source_bytes, source_name = await _read_upload(source)
    target_bytes, target_name = await _read_upload(target)
    if len(source_bytes) + len(target_bytes) > MAX_BODY_BYTES:
        raise mapper.FileTooLargeError("The two files together are over the 4 MB limit.")
    source_table = _load(source_bytes, source_name, source_sheet, header_only=False)
    target_table = _load(target_bytes, target_name, target_sheet, header_only=True)
    rows = mapper.local_match(
        [str(column) for column in source_table.frame.columns],
        [str(column) for column in target_table.frame.columns],
        threshold=_threshold(threshold),
    )
    return {
        "mappings": [_mapping_payload(row) for row in rows],
        "choices": _column_choices(source_table),
        "target_labels": {
            str(column): target_table.originals.get(str(column), str(column))
            for column in target_table.frame.columns
        },
        "duplicate_rows": mapper.duplicate_row_count(source_table.frame),
        **_profiles(source_table, target_table),
    }


@app.post("/api/suggest")
async def suggest_headers(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    mappings: str = Form("[]"),
    source_sheet: str = Form(""),
    target_sheet: str = Form(""),
    headers_only: str = Form("0"),
) -> dict:
    source_bytes, source_name = await _read_upload(source)
    target_bytes, target_name = await _read_upload(target)
    if len(source_bytes) + len(target_bytes) > MAX_BODY_BYTES:
        raise mapper.FileTooLargeError("The two files together are over the 4 MB limit.")
    source_table = _load(source_bytes, source_name, source_sheet, header_only=False)
    target_table = _load(target_bytes, target_name, target_sheet, header_only=True)
    current = _mappings_from_client(mappings, source_table, target_table)
    unresolved = [row.target for row in current if row.source is None]
    used = {row.source for row in current if row.source}
    unused = [str(column) for column in source_table.frame.columns if column not in used]
    if not unresolved:
        return {"mappings": [_mapping_payload(row) for row in current], "message": "Every header already has a source column."}

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return {"mappings": [_mapping_payload(row) for row in current], "checked": False}

    payload = mapper.build_llm_payload(
        [mapper.profile_column(column, source_table.frame[column]) for column in unused],
        mapper.profile_headers(unresolved),
        headers_only=_flag(headers_only),
    )
    reply = _call_gemini(key, mapper.build_llm_prompt(payload))
    suggested = mapper.parse_and_validate_llm(
        reply,
        allowed_targets=unresolved,
        allowed_sources=unused,
        used_sources=used,
    )
    merged = mapper.merge_llm_mappings(current, suggested)
    return {
        "mappings": [_mapping_payload(row) for row in merged],
        "checked": True,
        "message": "Reviewed the remaining headers. Check every row marked Needs review.",
    }


@app.post("/api/convert")
async def convert_sheet(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    mappings: str = Form("[]"),
    source_sheet: str = Form(""),
    target_sheet: str = Form(""),
    drop_duplicates: str = Form("0"),
    file_format: str = Form("csv"),
    preview: str = Form("0"),
) -> Response:
    source_bytes, source_name = await _read_upload(source)
    target_bytes, target_name = await _read_upload(target)
    if len(source_bytes) + len(target_bytes) > MAX_BODY_BYTES:
        raise mapper.FileTooLargeError("The two files together are over the 4 MB limit.")
    source_table = _load(source_bytes, source_name, source_sheet, header_only=False)
    target_table = _load(target_bytes, target_name, target_sheet, header_only=True)
    rows = _mappings_from_client(mappings, source_table, target_table)
    output, summary = mapper.convert_frame(
        source_table.frame,
        [str(column) for column in target_table.frame.columns],
        target_table.originals,
        rows,
        drop_duplicate_rows=_flag(drop_duplicates),
    )
    summary_payload = {
        "columns_mapped": summary.columns_mapped,
        "unmapped_targets": summary.unmapped_targets,
        "rows_in": summary.rows_in,
        "rows_kept": summary.rows_kept,
        "rows_removed": summary.rows_removed,
        "duplicate_rows": summary.duplicate_rows,
        "preview": _table(output.fillna("")),
        "before": _table(source_table.frame),
    }
    if _flag(preview):
        return JSONResponse(summary_payload)

    kind = file_format.strip().lower()
    if kind == "xlsx":
        body = mapper.to_xlsx_bytes(output)
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "mapped.xlsx"
    else:
        body = mapper.to_csv_bytes(output)
        media = "text/csv"
        filename = "mapped.csv"
    return Response(
        content=body,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Rows-Kept": str(summary.rows_kept),
            "X-Rows-Removed": str(summary.rows_removed),
        },
    )


def _call_gemini(api_key: str, prompt: str) -> str:
    """Ask Gemini to map the remaining headers. The key and prompt are not logged."""

    import httpx

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    if not MODEL_RE.fullmatch(model):
        raise mapper.MapperError("GEMINI_MODEL is not a valid model id.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": mapper.SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096},
    }
    try:
        response = httpx.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=body,
            timeout=45.0,
        )
    except httpx.HTTPError as exc:
        raise mapper.MapperError("Could not reach Gemini. Retry in a moment.") from exc
    if response.status_code in {401, 403}:
        raise mapper.MapperError("Gemini rejected the API key.")
    if response.status_code >= 400:
        raise mapper.MapperError(f"Gemini returned status {response.status_code}. Retry in a moment.")
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise mapper.LLMResponseError("Gemini returned an unreadable response. Retry the request.") from exc
    candidates = data.get("candidates") if isinstance(data, dict) else None
    parts: list[str] = []
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        content = candidates[0].get("content")
        raw_parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(raw_parts, list):
            parts = [part.get("text", "") for part in raw_parts if isinstance(part, dict) and part.get("text")]
    if not parts:
        raise mapper.LLMResponseError("Gemini returned an empty response. Retry the request.")
    return "\n".join(parts)


def _page_bytes() -> bytes:
    """Page bytes shipped with the function.

    On Vercel the public directory is served by the CDN and is not inside the
    function, so a bad static file would hide this route. The deployment omits
    public/index.html and includes page.part* next to this module instead.
    Locally those parts are absent and the public page is used.
    """
    root = Path(__file__).resolve().parent
    parts = sorted(root.glob("page.part*"))
    if parts:
        return b"".join(path.read_bytes() for path in parts)
    return (PUBLIC / "index.html").read_bytes()


@app.get("/")
def home() -> Response:
    return Response(_page_bytes(), media_type="text/html; charset=utf-8")
