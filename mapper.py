"""Header mapping, PII masking, and sheet conversion.

This module is UI-free. Uploaded files are read from memory and nothing here
writes sheet contents to disk or to a log.

Privacy rule: a column is sensitive when its header contains a sensitive token
or at least 30% of sampled values match a PII pattern. Sensitive columns
contribute irreversible shape patterns only (A upper, a lower, 9 digit).
Every other column contributes at most three samples, with inline PII replaced
by tags such as [EMAIL]. Unclear matches are left empty rather than guessed.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import math
import re
from dataclasses import dataclass

import pandas as pd
from openpyxl import load_workbook

# --- limits and labels -------------------------------------------------------

SAMPLE_ROWS = 200
MAX_SAMPLES = 3
MAX_SAMPLE_CHARS = 80
SHAPE_LIMIT = 48
PII_RATIO = 0.30
DEFAULT_FUZZY_THRESHOLD = 0.90
LOW_CONFIDENCE = 0.60
MATCHED_CONFIDENCE = 0.90
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
CLEAR = "— Clear —"

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

SENSITIVE_TOKENS = (
    "email",
    "phone",
    "mobile",
    "name",
    "address",
    "dob",
    "aadhaar",
    "pan",
    "passport",
    "ssn",
    "card",
    "account",
    "iban",
    "ifsc",
    "salary",
    "password",
    "token",
)

# Short tokens are matched as whole words so "company" is not "pan"
# and "filename" is not "name".
_SHORT_TOKENS = {"name", "dob", "pan", "ssn", "card", "iban", "ifsc", "token"}
_NAME_COMPOUNDS = {
    "fullname",
    "firstname",
    "lastname",
    "middlename",
    "surname",
    "givenname",
    "familyname",
    "username",
    "customername",
    "clientname",
    "employeename",
    "personname",
    "displayname",
    "legalname",
    "nickname",
    "empname",
}

SYSTEM_PROMPT = (
    "You map spreadsheet column headers. Return only a JSON array, with no "
    "markdown. Each object has the keys target, source, confidence, and reason. "
    "source is an exact source header from the payload, or null. Use each source "
    "at most once. If you are not sure, set source to null and confidence to 0. "
    "Never invent a header. Samples may be masked or reduced to shape patterns; "
    "do not try to reconstruct the hidden characters."
)


class MapperError(Exception):
    """A problem the user can act on. The message is safe to show in the UI."""


class EncodingError(MapperError):
    pass


class EmptyFileError(MapperError):
    pass


class MissingHeaderError(MapperError):
    pass


class FileTooLargeError(MapperError):
    pass


class LLMResponseError(MapperError):
    pass


@dataclass
class LoadedTable:
    """A sheet kept in memory. Columns are unique working names."""

    frame: pd.DataFrame
    originals: dict[str, str]
    duplicate_names: list[str]
    row_count: int
    sheet_name: str | None = None


@dataclass
class ColumnProfile:
    header: str
    inferred_type: str
    sensitive: bool
    reasons: list[str]
    transmitted: list[str]
    transmission: str  # "patterns", "masked_samples", or "none"


@dataclass
class Mapping:
    target: str
    source: str | None
    confidence: float
    reason: str
    status: str
    origin: str


@dataclass
class ConvertSummary:
    columns_mapped: int
    unmapped_targets: int
    rows_in: int
    rows_kept: int
    rows_removed: int


# --- headers -----------------------------------------------------------------

def normalize_header(name: str) -> str:
    """Lowercase and drop punctuation and spaces so 'E-mail' matches 'email'."""

    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def uniquify_headers(headers: list[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """Suffix duplicate headers as name, name__2, name__3.

    The returned map sends each working name back to the original text so the
    output file can restore it. `duplicate_names` lists each repeated original
    once, in first-seen order.
    """

    cleaned = [_clean_header(header) for header in headers]
    totals: dict[str, int] = {}
    for header in cleaned:
        totals[header] = totals.get(header, 0) + 1

    seen_count: dict[str, int] = {}
    used: set[str] = set()
    working: list[str] = []
    originals: dict[str, str] = {}
    duplicates: list[str] = []
    already_listed: set[str] = set()

    for header in cleaned:
        seen_count[header] = seen_count.get(header, 0) + 1
        occurrence = seen_count[header]
        if totals[header] == 1:
            candidate = header
        else:
            candidate = header if occurrence == 1 else f"{header}__{occurrence}"
            if header not in already_listed:
                duplicates.append(header)
                already_listed.add(header)
        while candidate in used:
            occurrence += 1
            candidate = f"{header}__{occurrence}"
        used.add(candidate)
        working.append(candidate)
        originals[candidate] = header
    return working, originals, duplicates


def _clean_header(header: object) -> str:
    if header is None:
        return "column"
    try:
        if not isinstance(header, str) and pd.isna(header):
            return "column"
    except TypeError:
        pass
    text = str(header).strip()
    if not text or text.lower() == "nan":
        return "column"
    return text


def header_sensitivity_hits(header: str) -> list[str]:
    """Sensitive tokens present in a header, in catalogue order.

    Hyphens are removed inside a word so "E-mail" counts as "email", while
    "filename" does not count as "name".
    """

    lowered = str(header).lower()
    split_tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    word_tokens = []
    for word in re.split(r"[\s_./]+", lowered):
        squashed = re.sub(r"[^a-z0-9]+", "", word)
        if squashed:
            word_tokens.append(squashed)
    tokens = split_tokens + word_tokens
    extra = _compound_token_hits(tokens)
    hits: list[str] = []
    for token in SENSITIVE_TOKENS:
        if token in tokens or token in extra:
            hits.append(token)
    return hits


def _compound_token_hits(tokens: list[str]) -> set[str]:
    found: set[str] = set()
    for token in tokens:
        if token in _NAME_COMPOUNDS:
            found.add("name")
        for sensitive in SENSITIVE_TOKENS:
            if sensitive in _SHORT_TOKENS or sensitive == token:
                continue
            if len(sensitive) >= 5 and sensitive in token:
                found.add(sensitive)
    return found


# --- masking -----------------------------------------------------------------

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CARD = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b|\b\d{13,19}\b")
_AADHAAR = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_PASSPORT = re.compile(r"\b[A-Z][0-9]{7}\b")
_PHONE = re.compile(
    r"(?:\+\d{1,3}[\s\-.]?)?(?:\(\d{3}\)[\s\-.]?\d{3}[\s\-.]?\d{4}|\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4})|\b[6-9]\d{9}\b"
)

# Longer and more specific patterns run first so a card is not tagged as a phone.
_INLINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("[EMAIL]", _EMAIL),
    ("[IBAN]", _IBAN),
    ("[CARD]", _CARD),
    ("[AADHAAR]", _AADHAAR),
    ("[SSN]", _SSN),
    ("[IFSC]", _IFSC),
    ("[PAN]", _PAN),
    ("[PASSPORT]", _PASSPORT),
    ("[PHONE]", _PHONE),
)

_FULL_PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\Z"),
    re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}\Z"),
    re.compile(r"(?:\d{4}[ -]?){3}\d{4}\Z|\d{13,19}\Z"),
    re.compile(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}\Z"),
    re.compile(r"\d{3}-\d{2}-\d{4}\Z"),
    re.compile(r"[A-Z]{4}0[A-Z0-9]{6}\Z"),
    re.compile(r"[A-Z]{5}\d{4}[A-Z]\Z"),
    re.compile(r"[A-Z][0-9]{7}\Z"),
    re.compile(r"(?:\+?\d{1,3}[\s\-.]?)?(?:\(\d{3}\)[\s\-.]?)?\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4}\Z"),
    re.compile(r"[6-9]\d{9}\Z"),
)


def to_shape(value: object, limit: int = SHAPE_LIMIT) -> str:
    """Replace letters and digits so the original value cannot be recovered."""

    text = _as_text(value)
    if text is None:
        return ""
    chars: list[str] = []
    for char in text[:limit]:
        if char.isupper():
            chars.append("A")
        elif char.islower():
            chars.append("a")
        elif char.isdigit():
            chars.append("9")
        elif char.isspace():
            chars.append(" ")
        else:
            chars.append(char)
    if len(text) > limit:
        chars.append("…")
    return "".join(chars)


def mask_inline_pii(text: str) -> str:
    """Replace PII inside an otherwise ordinary sample. Non-PII words stay."""

    masked = str(text)
    for tag, pattern in _INLINE_PATTERNS:
        masked = pattern.sub(tag, masked)
    return masked


def value_matches_pii(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return any(pattern.fullmatch(text) for pattern in _FULL_PATTERNS)


def profile_column(
    header: str,
    series: pd.Series | None,
    *,
    sample_rows: int = SAMPLE_ROWS,
) -> ColumnProfile:
    """Describe one column and the exact strings that may be sent to a model."""

    hits = header_sensitivity_hits(header)
    reasons = [f"Header contains '{hit}'." for hit in hits]
    samples = _sample_strings(series, sample_rows) if series is not None else []
    if series is not None:
        checked = _sample_strings(series, sample_rows, limit=sample_rows)
        if checked:
            flagged = sum(1 for value in checked if value_matches_pii(value))
            ratio = flagged / len(checked)
            if ratio >= PII_RATIO:
                reasons.append(f"{ratio:.0%} of sampled values match a PII pattern.")
    sensitive = bool(reasons)
    if not samples:
        transmitted: list[str] = []
        transmission = "none"
    elif sensitive:
        transmitted = [to_shape(value) for value in samples]
        transmission = "patterns"
    else:
        transmitted = [mask_inline_pii(value)[:MAX_SAMPLE_CHARS] for value in samples]
        transmission = "masked_samples"
    return ColumnProfile(
        header=str(header),
        inferred_type=infer_series_type(series),
        sensitive=sensitive,
        reasons=reasons,
        transmitted=transmitted,
        transmission=transmission,
    )


def profile_frame(frame: pd.DataFrame, *, sample_rows: int = SAMPLE_ROWS) -> list[ColumnProfile]:
    return [
        profile_column(str(column), frame[column], sample_rows=sample_rows)
        for column in frame.columns
    ]


def profile_headers(headers: list[str]) -> list[ColumnProfile]:
    """Profiles for a target sheet. No values are available, so none are sent."""

    return [profile_column(header, None) for header in headers]


def build_llm_payload(
    source_profiles: list[ColumnProfile],
    target_profiles: list[ColumnProfile],
    *,
    headers_only: bool,
) -> dict:
    """JSON-ready payload. This is the only sheet content a model is allowed to see."""

    return {
        "headers_only": headers_only,
        "targets": [_pack_profile(profile, headers_only=headers_only) for profile in target_profiles],
        "sources": [_pack_profile(profile, headers_only=headers_only) for profile in source_profiles],
    }


def payload_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_llm_prompt(payload: dict) -> str:
    return (
        "Map every target header to at most one source header from the list.\n"
        "Return a JSON array of "
        '{"target": "...", "source": "... or null", "confidence": 0.0, "reason": "..."}.\n'
        "If a mapping is unclear, source must be null. Do not guess.\n\n"
        f"{payload_text(payload)}"
    )


def _pack_profile(profile: ColumnProfile, *, headers_only: bool) -> dict:
    item: dict = {
        "header": profile.header,
        "type": profile.inferred_type,
        "sensitive": profile.sensitive,
    }
    if headers_only or profile.transmission == "none":
        return item
    if profile.transmission == "patterns":
        item["patterns"] = list(profile.transmitted)
    elif profile.transmitted:
        item["masked_samples"] = list(profile.transmitted)
    return item


def _sample_strings(
    series: pd.Series | None,
    sample_rows: int,
    *,
    limit: int = MAX_SAMPLES,
) -> list[str]:
    if series is None:
        return []
    values: list[str] = []
    for value in series.head(sample_rows).tolist():
        text = _as_text(value)
        if text is None:
            continue
        stripped = text.strip()
        if not stripped:
            continue
        values.append(stripped)
        if len(values) >= limit:
            break
    return values


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def infer_series_type(series: pd.Series | None) -> str:
    if series is None:
        return "unknown"
    values = series.dropna()
    if values.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(values):
        return "boolean"
    if pd.api.types.is_integer_dtype(values):
        return "integer"
    if pd.api.types.is_numeric_dtype(values):
        return "number"
    sample = values.astype(str).str.strip().head(50)
    sample = sample[sample != ""]
    if sample.empty:
        return "empty"
    if sample.map(_looks_like_date).mean() >= 0.8:
        return "date"
    numeric = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return "number"
    return "text"


def _looks_like_date(value: str) -> bool:
    if not re.search(r"[-/]|[A-Za-z]", value):
        return False
    parsed = pd.to_datetime(value, errors="coerce")
    return not pd.isna(parsed)


# --- matching ----------------------------------------------------------------

def similarity(left: str, right: str) -> float:
    """Full-string similarity in 0..1. Partial scores are avoided on purpose."""

    if not left or not right:
        return 0.0
    try:
        from rapidfuzz.fuzz import ratio
    except ImportError:
        return difflib.SequenceMatcher(None, left, right).ratio()
    return float(ratio(left, right)) / 100.0


def local_match(
    source_headers: list[str],
    target_headers: list[str],
    *,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[Mapping]:
    """Exact matches first, then fuzzy matches at or above `threshold`.

    A source is used at most once. Ties and below-threshold scores stay empty.
    """

    threshold = float(threshold)
    source_norm = {header: normalize_header(header) for header in source_headers}
    by_norm: dict[str, list[str]] = {}
    for header, norm in source_norm.items():
        by_norm.setdefault(norm, []).append(header)

    used: set[str] = set()
    assigned: dict[str, Mapping] = {}
    pending: list[str] = []

    for target in target_headers:
        candidates = [header for header in by_norm.get(normalize_header(target), []) if header not in used]
        if len(candidates) == 1:
            source = candidates[0]
            used.add(source)
            assigned[target] = Mapping(
                target=target,
                source=source,
                confidence=1.0,
                reason="Exact match after normalising case, punctuation, and spaces.",
                status="Matched",
                origin="exact",
            )
        elif len(candidates) > 1:
            assigned[target] = Mapping(
                target=target,
                source=None,
                confidence=0.0,
                reason=(
                    "Several source columns normalise to this header ("
                    + ", ".join(candidates)
                    + "). Not auto-assigned."
                ),
                status="Needs review",
                origin="none",
            )
        else:
            pending.append(target)

    for target in pending:
        norm = normalize_header(target)
        scored: list[tuple[float, str]] = []
        for header, header_norm in source_norm.items():
            if header in used:
                continue
            score = similarity(norm, header_norm)
            if score >= threshold:
                scored.append((score, header))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored:
            assigned[target] = Mapping(
                target=target,
                source=None,
                confidence=0.0,
                reason="No local match above the threshold.",
                status="Unmapped",
                origin="none",
            )
            continue
        best_score, best_source = scored[0]
        second = scored[1][0] if len(scored) > 1 else None
        if second is not None and best_score - second < 0.03:
            names = ", ".join(header for _, header in scored[:4])
            assigned[target] = Mapping(
                target=target,
                source=None,
                confidence=round(best_score, 4),
                reason=f"Close match with {names}. Not auto-assigned.",
                status="Needs review",
                origin="none",
            )
            continue
        used.add(best_source)
        assigned[target] = Mapping(
            target=target,
            source=best_source,
            confidence=round(best_score, 4),
            reason=f"Fuzzy match ({best_score:.2f}) after normalising headers.",
            status="Matched" if best_score >= MATCHED_CONFIDENCE else "Needs review",
            origin="fuzzy",
        )
    return [assigned[target] for target in target_headers]


def rematch(
    source_headers: list[str],
    target_headers: list[str],
    existing: list[Mapping] | None,
    *,
    threshold: float,
) -> list[Mapping]:
    """Re-run local matching without replacing rows the user set by hand."""

    fresh = local_match(source_headers, target_headers, threshold=threshold)
    if not existing:
        return fresh
    manual = {row.target: row for row in existing if row.origin == "manual"}
    used = {row.source for row in manual.values() if row.source}
    output: list[Mapping] = []
    for row in fresh:
        kept = manual.get(row.target)
        if kept is not None:
            output.append(kept)
            continue
        if row.source is not None and row.source in used:
            output.append(
                Mapping(
                    target=row.target,
                    source=None,
                    confidence=row.confidence,
                    reason=f"'{row.source}' is already used by a manual mapping.",
                    status="Needs review",
                    origin="none",
                )
            )
            continue
        if row.source:
            used.add(row.source)
        output.append(row)
    return output


def apply_manual_choice(
    mappings: list[Mapping],
    target: str,
    choice: str | None,
) -> str | None:
    """Assign `choice` to `target`. Returns an error when the source is taken."""

    row = next((item for item in mappings if item.target == target), None)
    if row is None:
        raise MapperError(f"Unknown target header '{target}'.")
    if choice is None or choice.strip() == "" or choice == CLEAR:
        row.source = None
        row.confidence = 0.0
        row.reason = "Cleared."
        row.status = "Unmapped"
        row.origin = "manual"
        return None
    for other in mappings:
        if other.target != target and other.source == choice:
            return (
                f"'{choice}' is already mapped to '{other.target}'. "
                "Clear that row, then assign it here."
            )
    row.source = choice
    row.confidence = 1.0
    row.reason = "Chosen manually."
    row.status = "Matched"
    row.origin = "manual"
    return None


def parse_and_validate_llm(
    text: str,
    *,
    allowed_targets: list[str],
    allowed_sources: list[str],
    used_sources: set[str] | None = None,
) -> list[Mapping]:
    """Parse a model reply and drop anything that is not a real, unused column.

    Confidence under 0.60 is treated as a guess: the source is cleared and the
    row is flagged Needs review. Unknown targets are ignored.
    """

    items = _extract_json_array(text)
    allowed = set(allowed_sources)
    used = set(used_sources or ())
    grouped: dict[str, list[dict]] = {target: [] for target in allowed_targets}
    for item in items:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        if isinstance(target, str) and target in grouped:
            grouped[target].append(item)

    results: list[Mapping] = []
    for target in allowed_targets:
        candidates = grouped[target]
        if not candidates:
            results.append(
                Mapping(
                    target=target,
                    source=None,
                    confidence=0.0,
                    reason="The model did not return this header.",
                    status="Unmapped",
                    origin="llm",
                )
            )
            continue
        chosen = max(candidates, key=_confidence_of)
        source, confidence, reason, status = _validate_llm_item(chosen, allowed, used)
        if source is not None:
            used.add(source)
        results.append(
            Mapping(
                target=target,
                source=source,
                confidence=confidence,
                reason=reason,
                status=status,
                origin="llm",
            )
        )
    return results


def merge_llm_mappings(current: list[Mapping], suggested: list[Mapping]) -> list[Mapping]:
    """Fill empty, non-manual rows. Manual choices and existing sources stay."""

    by_target = {row.target: row for row in suggested}
    merged: list[Mapping] = []
    for row in current:
        if row.origin == "manual" or row.source is not None:
            merged.append(row)
            continue
        merged.append(by_target.get(row.target, row))
    return merged


def _extract_json_array(text: str) -> list:
    raw = (text or "").strip()
    fenced = re.sub(r"^```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```\s*$", "", fenced, count=1)
    parsed = None
    for candidate in (fenced, raw):
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if match is None:
            raise LLMResponseError("The model did not return a JSON array. Retry the request.")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMResponseError("The model did not return a JSON array. Retry the request.") from exc
    if isinstance(parsed, dict):
        for key in ("mappings", "result", "columns"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise LLMResponseError("The model did not return a JSON array. Retry the request.")
    return parsed


def _validate_llm_item(
    item: dict,
    allowed_sources: set[str],
    used_sources: set[str],
) -> tuple[str | None, float, str, str]:
    reason = str(item.get("reason") or "").strip()[:240]
    confidence = _confidence_of(item)
    source = _source_of(item)
    if source is None:
        return None, confidence, reason or "Unclear. Not auto-assigned.", "Needs review"
    if source not in allowed_sources:
        return (
            None,
            0.0,
            f"Model returned an unknown column '{source}'. Cleared.",
            "Needs review",
        )
    if source in used_sources:
        return (
            None,
            confidence,
            f"'{source}' is already used. Not auto-assigned.",
            "Needs review",
        )
    if confidence < LOW_CONFIDENCE:
        extra = "Confidence is too low to assign. Not auto-assigned."
        return None, confidence, f"{reason} {extra}".strip(), "Needs review"
    status = "Matched" if confidence >= MATCHED_CONFIDENCE else "Needs review"
    return source, confidence, reason or "Suggested by the model.", status


def _source_of(item: dict) -> str | None:
    source = item.get("source", None)
    if source is None:
        return None
    if not isinstance(source, str):
        return None
    cleaned = source.strip()
    if cleaned.lower() in {"", "null", "none"}:
        return None
    return cleaned


def _confidence_of(item: dict) -> float:
    try:
        value = float(item.get("confidence", 0))
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    if value > 1:
        value = value / 100.0
    return max(0.0, min(1.0, value))


# --- duplicates and conversion ----------------------------------------------

def find_identical_columns(frame: pd.DataFrame) -> list[list[str]]:
    """Groups of columns whose values are exactly the same, including blanks."""

    columns = [str(column) for column in frame.columns]
    seen: set[str] = set()
    groups: list[list[str]] = []
    for index, left in enumerate(columns):
        if left in seen:
            continue
        group = [left]
        for right in columns[index + 1 :]:
            if right in seen:
                continue
            if frame[left].equals(frame[right]):
                group.append(right)
                seen.add(right)
        if len(group) > 1:
            seen.add(left)
            groups.append(group)
    return groups


def duplicate_row_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame.duplicated().sum())


def convert_frame(
    source: pd.DataFrame,
    target_working: list[str],
    target_originals: dict[str, str],
    mappings: list[Mapping],
    *,
    drop_duplicate_rows: bool,
) -> tuple[pd.DataFrame, ConvertSummary]:
    """Build a sheet in target order. Missing mappings become blank columns.

    Output headers are the target's original names, so a suffixed working name
    such as Amount__2 is written as Amount again.
    """

    by_target = {row.target: row for row in mappings}
    table = source.reset_index(drop=True)
    series_list: list[pd.Series] = []
    names: list[str] = []
    mapped = 0
    unmapped = 0
    for working in target_working:
        row = by_target.get(working)
        source_name = row.source if row is not None else None
        if source_name and source_name in table.columns:
            series_list.append(table[source_name].reset_index(drop=True))
            mapped += 1
        else:
            series_list.append(pd.Series([pd.NA] * len(table), dtype="object"))
            unmapped += 1
        names.append(target_originals.get(working, working))

    if series_list:
        output = pd.concat(series_list, axis=1, ignore_index=True)
        output.columns = names
    else:
        output = pd.DataFrame(columns=names)

    rows_in = int(len(output))
    rows_removed = 0
    if drop_duplicate_rows and not output.empty:
        before = len(output)
        output = output.drop_duplicates().reset_index(drop=True)
        rows_removed = int(before - len(output))
    summary = ConvertSummary(
        columns_mapped=mapped,
        unmapped_targets=unmapped,
        rows_in=rows_in,
        rows_kept=int(len(output)),
        rows_removed=rows_removed,
    )
    return output, summary


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


# --- loading -----------------------------------------------------------------

def list_sheets(data: bytes, filename: str) -> list[str]:
    """Worksheet names for an Excel file. CSV files have none."""

    if not _is_excel(data, filename):
        return []
    _require_zip(data)
    workbook = _open_workbook(data)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()


def load_table(
    data: bytes,
    filename: str,
    *,
    sheet: str | None = None,
    header_only: bool = False,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> LoadedTable:
    """Load a CSV or xlsx workbook from bytes.

    `header_only` reads the first row and does not put any data cells into the
    returned frame. Profiling of a full load still samples later; this function
    keeps the source rows because conversion needs them. Files over `max_bytes`
    are rejected before parsing.
    """

    if data is None or len(data) == 0 or not data.strip():
        raise EmptyFileError("The file is empty.")
    if len(data) > max_bytes:
        limit_mb = max(1, max_bytes // (1024 * 1024))
        raise FileTooLargeError(
            f"This file is over the {limit_mb} MB limit. Split it, or upload a smaller sheet."
        )
    if filename.lower().endswith(".xls") and not filename.lower().endswith(".xlsx"):
        raise MapperError("Old .xls workbooks are not supported. Save the file as .xlsx and upload it again.")

    if _is_excel(data, filename):
        return _load_excel(data, sheet=sheet, header_only=header_only)
    return _load_csv(data, header_only=header_only)


def _load_csv(data: bytes, *, header_only: bool) -> LoadedTable:
    raw_headers, encoding = _csv_headers(data)
    _reject_missing_header(raw_headers)
    working, originals, duplicates = uniquify_headers(raw_headers)
    if header_only:
        frame = pd.DataFrame(columns=working)
        return LoadedTable(frame, originals, duplicates, row_count=0, sheet_name=None)
    try:
        frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
    except pd.errors.EmptyDataError as exc:
        raise EmptyFileError("The file has no rows.") from exc
    except pd.errors.ParserError as exc:
        raise MapperError("The CSV could not be parsed. Check that the first row is a header.") from exc
    frame = _align_frame(frame, working, data, encoding, excel_sheet=None)
    return LoadedTable(frame, originals, duplicates, row_count=int(len(frame)), sheet_name=None)


def _load_excel(data: bytes, *, sheet: str | None, header_only: bool) -> LoadedTable:
    _require_zip(data)
    workbook = _open_workbook(data)
    try:
        names = list(workbook.sheetnames)
        if not names:
            raise EmptyFileError("The workbook has no sheets.")
        selected = sheet or names[0]
        if selected not in names:
            raise MapperError(f"Sheet '{selected}' was not found. Choose one of: {', '.join(names)}.")
        worksheet = workbook[selected]
        first = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True), None)
    finally:
        workbook.close()
    if first is None:
        raise EmptyFileError("The sheet is empty.")
    raw_headers = ["" if cell is None else str(cell).strip() for cell in first]
    _reject_missing_header(raw_headers)
    working, originals, duplicates = uniquify_headers(raw_headers)
    if header_only:
        # Deliberately do not read row 2 or below into a dataframe.
        frame = pd.DataFrame(columns=working)
        return LoadedTable(frame, originals, duplicates, row_count=0, sheet_name=selected)
    try:
        frame = pd.read_excel(io.BytesIO(data), sheet_name=selected, engine="openpyxl")
    except Exception as exc:
        raise MapperError("The worksheet could not be read. Check that the first row is a header.") from exc
    frame = _align_frame(frame, working, data, encoding=None, excel_sheet=selected)
    return LoadedTable(frame, originals, duplicates, row_count=int(len(frame)), sheet_name=selected)


def _align_frame(
    frame: pd.DataFrame,
    working: list[str],
    data: bytes,
    encoding: str | None,
    excel_sheet: str | None,
) -> pd.DataFrame:
    if len(frame.columns) == len(working):
        aligned = frame.copy()
        aligned.columns = working
        return aligned
    # Pandas renames duplicate headers before we see them. Re-read by position.
    if excel_sheet is not None:
        positioned = pd.read_excel(
            io.BytesIO(data),
            sheet_name=excel_sheet,
            engine="openpyxl",
            header=None,
            skiprows=1,
        )
    else:
        positioned = pd.read_csv(
            io.BytesIO(data),
            encoding=encoding or "utf-8",
            header=None,
            skiprows=1,
        )
    positioned = positioned.iloc[:, : len(working)]
    if positioned.shape[1] < len(working):
        for _ in range(len(working) - positioned.shape[1]):
            positioned[positioned.shape[1]] = pd.NA
    positioned.columns = working
    return positioned


def _csv_headers(data: bytes) -> tuple[list[str], str]:
    last_error: Exception | None = None
    saw_decode = False
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
            saw_decode = True
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise EmptyFileError("The file has no rows.") from exc
        return header, encoding
    if saw_decode:
        raise MapperError("The CSV could not be parsed. Check that the first row is a header.")
    raise EncodingError("Could not read this CSV. Save it as UTF-8 or Excel and try again.") from last_error


def _reject_missing_header(headers: list[str]) -> None:
    cleaned = [str(header).strip() for header in headers]
    if not cleaned or all(not header for header in cleaned):
        raise MissingHeaderError(
            "This sheet is missing a header row. Put column names in the first row and upload it again."
        )
    unnamed = [header for header in cleaned if header.lower().startswith("unnamed")]
    if unnamed and len(unnamed) == len(cleaned):
        raise MissingHeaderError(
            "This sheet is missing a header row. Put column names in the first row and upload it again."
        )


def _is_excel(data: bytes, filename: str) -> bool:
    name = filename.lower()
    if name.endswith(".xlsx"):
        return True
    if name.endswith(".csv"):
        return False
    return data.startswith(b"PK")


def _require_zip(data: bytes) -> None:
    if not data.startswith(b"PK"):
        raise MapperError("This file does not look like an Excel workbook. Upload a .xlsx or a .csv file.")


def _open_workbook(data: bytes):
    try:
        return load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise MapperError("Could not open this workbook. Check that it is a valid .xlsx file.") from exc

