"""Local matching and validation of the model's JSON reply."""

from __future__ import annotations

import mapper


def test_normalise_strips_case_punctuation_and_spaces() -> None:
    assert mapper.normalize_header("  E-mail Address ") == "emailaddress"
    assert mapper.normalize_header("First_Name") == "firstname"


def test_exact_match_after_normalising() -> None:
    rows = mapper.local_match(
        ["First Name", "E-mail", "Order ID"],
        ["first_name", "email", "order_id"],
    )
    by_target = {row.target: row for row in rows}
    assert by_target["first_name"].source == "First Name"
    assert by_target["email"].source == "E-mail"
    assert by_target["order_id"].source == "Order ID"
    assert all(row.status == "Matched" and row.confidence == 1.0 for row in rows)
    assert all(row.origin == "exact" for row in rows)


def test_fuzzy_match_above_threshold() -> None:
    rows = mapper.local_match(["Custmer Name"], ["Customer Name"], threshold=0.90)
    assert rows[0].source == "Custmer Name"
    assert rows[0].origin == "fuzzy"
    assert rows[0].confidence >= 0.90
    assert rows[0].status == "Matched"


def test_fuzzy_below_threshold_does_not_guess() -> None:
    rows = mapper.local_match(["Amt"], ["amount"], threshold=0.90)
    assert rows[0].source is None
    assert rows[0].status == "Unmapped"


def test_ambiguous_fuzzy_match_stays_unassigned() -> None:
    rows = mapper.local_match(["address1", "address2"], ["address"], threshold=0.90)
    assert rows[0].source is None
    assert rows[0].status == "Needs review"
    assert "address1" in rows[0].reason
    assert "address2" in rows[0].reason


def test_ambiguous_exact_normalisation_does_not_guess() -> None:
    rows = mapper.local_match(["Email", "E-mail"], ["email_address"])
    # Neither source normalises to emailaddress, so this is not the exact-tie case.
    # Two sources that DO normalise to the same target are the tie:
    tied = mapper.local_match(["Email", "E mail"], ["email"])
    assert tied[0].source is None
    assert tied[0].status == "Needs review"


def test_lowered_threshold_can_suggest_a_review_row() -> None:
    rows = mapper.local_match(["postalcode"], ["postcode"], threshold=0.80)
    assert rows[0].source == "postalcode"
    assert rows[0].status == "Needs review"
    assert 0.80 <= rows[0].confidence < 0.90


def test_llm_reply_is_validated() -> None:
    text = """```json
[
  {"target": "amount", "source": "Amt", "confidence": 0.82, "reason": "abbreviation"},
  {"target": "amount", "source": "Notes", "confidence": 0.4, "reason": "weaker duplicate"},
  {"target": "ghost", "source": "Amt", "confidence": 0.99, "reason": "unknown target"},
  {"target": "coupon", "source": "NotAColumn", "confidence": 0.95, "reason": "invented"},
  {"target": "notes", "source": null, "confidence": 0, "reason": "unclear"},
  {"target": "city", "source": "Amt", "confidence": 0.91, "reason": "already used"},
  {"target": "sku", "source": "Code", "confidence": 0.4, "reason": "too unsure"}
]
```"""
    rows = mapper.parse_and_validate_llm(
        text,
        allowed_targets=["amount", "coupon", "notes", "city", "sku"],
        allowed_sources=["Amt", "Notes", "Code"],
    )
    by_target = {row.target: row for row in rows}
    assert set(by_target) == {"amount", "coupon", "notes", "city", "sku"}
    assert by_target["amount"].source == "Amt"
    assert by_target["amount"].status == "Needs review"
    assert by_target["amount"].confidence == 0.82
    assert by_target["coupon"].source is None
    assert by_target["coupon"].status == "Needs review"
    assert "unknown" in by_target["coupon"].reason.lower() or "invent" in by_target["coupon"].reason.lower() or "NotAColumn" in by_target["coupon"].reason
    assert by_target["notes"].source is None
    assert by_target["notes"].status == "Needs review"
    assert by_target["city"].source is None
    assert by_target["sku"].source is None
    assert by_target["sku"].status == "Needs review"
    assert "ghost" not in by_target


def test_llm_rejects_a_source_already_used_locally() -> None:
    rows = mapper.parse_and_validate_llm(
        '[{"target": "mobile", "source": "Phone", "confidence": 0.95, "reason": "same idea"}]',
        allowed_targets=["mobile"],
        allowed_sources=["Phone", "Mobile"],
        used_sources={"Phone"},
    )
    assert rows[0].source is None
    assert rows[0].status == "Needs review"


def test_llm_garbage_raises_a_clear_error() -> None:
    try:
        mapper.parse_and_validate_llm(
            "I could not decide.",
            allowed_targets=["a"],
            allowed_sources=["b"],
        )
    except mapper.LLMResponseError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("expected LLMResponseError")


def test_merge_does_not_overwrite_manual_rows() -> None:
    current = [
        mapper.Mapping("email", "Inbox", 1.0, "Chosen manually.", "Matched", "manual"),
        mapper.Mapping("city", None, 0.0, "No local match.", "Unmapped", "none"),
    ]
    suggested = [
        mapper.Mapping("email", "Email", 0.95, "from model", "Matched", "llm"),
        mapper.Mapping("city", "Town", 0.88, "from model", "Needs review", "llm"),
    ]
    merged = mapper.merge_llm_mappings(current, suggested)
    assert merged[0].source == "Inbox"
    assert merged[0].origin == "manual"
    assert merged[1].source == "Town"
    assert merged[1].origin == "llm"
