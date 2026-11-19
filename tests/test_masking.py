"""Masking tests: raw PII must never appear in an LLM payload."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import mapper


RAW_EMAIL = "priya.shah@example.com"
RAW_EMAIL_2 = "arun.das@example.com"
RAW_PHONE = "9876543210"
RAW_NAME = "Priya Shah"
RAW_AADHAAR = "2345 6789 0123"
RAW_PAN = "ABCDE1234F"
RAW_SSN = "123-45-6789"
RAW_CARD = "4111111111111111"
RAW_IBAN = "GB82WEST12345698765432"
RAW_IFSC = "HDFC0001234"
RAW_ACCOUNT = "0099123456789012"
RAW_NOTE_EMAIL = "ping priya.shah@example.com tomorrow"
RAW_NOTE_PHONE = "call 9876543210 after 6"

FORBIDDEN = [
    RAW_EMAIL,
    RAW_EMAIL_2,
    RAW_PHONE,
    RAW_NAME,
    "Priya",
    "Shah",
    "example.com",
    "arun.das",
    RAW_AADHAAR,
    "234567890123",
    RAW_PAN,
    RAW_SSN,
    RAW_CARD,
    RAW_IBAN,
    RAW_IFSC,
    RAW_ACCOUNT,
    "009912345678",
]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Full Name": [RAW_NAME, "Arun Das", "Meera Iyer"],
            "Email": [RAW_EMAIL, RAW_EMAIL_2, "meera.iyer@example.com"],
            "Reference": [RAW_AADHAAR, "9876 5432 1098", "1111 2222 3333"],
            "PAN": [RAW_PAN, "BBBBB9999B", "CCCCC1111C"],
            "SSN": [RAW_SSN, "987-65-4321", "111-22-3333"],
            "Card": [RAW_CARD, "5500000000000004", "4000000000000002"],
            "IBAN": [RAW_IBAN, "DE89370400440532013000", "FR1420041010050500013M02606"],
            "IFSC": [RAW_IFSC, "SBIN0000456", "ICIC0000123"],
            "Account No": [RAW_ACCOUNT, "1111222233334444", "5555666677778888"],
            "Product": ["Widget", "Gadget", "Sprocket"],
            "Notes": [RAW_NOTE_EMAIL, "all good", RAW_NOTE_PHONE],
        }
    )


def _profiles(frame: pd.DataFrame) -> list[mapper.ColumnProfile]:
    return [mapper.profile_column(column, frame[column]) for column in frame.columns]


def _payload_text(headers_only: bool) -> str:
    frame = _frame()
    profiles = _profiles(frame)
    payload = mapper.build_llm_payload(profiles, profiles[:1], headers_only=headers_only)
    return mapper.payload_text(payload)


def test_shape_replaces_letters_and_digits() -> None:
    assert mapper.to_shape("Priya Shah-12") == "Aaaaa Aaaa-99"
    assert "Priya" not in mapper.to_shape(RAW_EMAIL)
    assert "12" not in mapper.to_shape("Priya Shah-12")
    assert set(mapper.to_shape("Ab9")) <= set("Aa9")


def test_inline_mask_replaces_pii_and_keeps_surrounding_words() -> None:
    masked = mapper.mask_inline_pii(RAW_NOTE_EMAIL)
    assert masked == "ping [EMAIL] tomorrow"
    assert RAW_EMAIL not in masked

    phone = mapper.mask_inline_pii(RAW_NOTE_PHONE)
    assert phone == "call [PHONE] after 6"
    assert RAW_PHONE not in phone

    mixed = mapper.mask_inline_pii(
        f"mail {RAW_EMAIL} pan {RAW_PAN} ssn {RAW_SSN} card {RAW_CARD} "
        f"aadhaar {RAW_AADHAAR} iban {RAW_IBAN} ifsc {RAW_IFSC}"
    )
    for secret in (RAW_EMAIL, RAW_PAN, RAW_SSN, RAW_CARD, RAW_AADHAAR, RAW_IBAN, RAW_IFSC):
        assert secret not in mixed
    for tag in ("[EMAIL]", "[PAN]", "[SSN]", "[CARD]", "[AADHAAR]", "[IBAN]", "[IFSC]"):
        assert tag in mixed


def test_sensitive_columns_transmit_shapes_only() -> None:
    frame = _frame()
    email = mapper.profile_column("Email", frame["Email"])
    assert email.sensitive
    assert email.transmission == "patterns"
    assert email.transmitted
    blob = " ".join(email.transmitted)
    assert RAW_EMAIL not in blob
    assert "example" not in blob.lower()
    assert "@" in blob
    assert any(char == "9" or char == "a" or char == "A" for char in blob)

    name = mapper.profile_column("Full Name", frame["Full Name"])
    assert name.sensitive
    assert "Priya" not in " ".join(name.transmitted)
    assert name.transmitted[0] == "Aaaaa Aaaa"


def test_pii_ratio_marks_a_neutral_header_sensitive() -> None:
    series = pd.Series([RAW_AADHAAR, "9876 5432 1098", "1111 2222 3333", "not an id"])
    profile = mapper.profile_column("Reference", series)
    assert profile.sensitive
    assert profile.transmission == "patterns"
    assert RAW_AADHAAR not in " ".join(profile.transmitted)
    assert "2345" not in " ".join(profile.transmitted)


def test_free_text_column_is_masked_inline_not_treated_as_raw() -> None:
    profile = mapper.profile_column("Notes", _frame()["Notes"])
    assert not profile.sensitive
    assert profile.transmission == "masked_samples"
    blob = " ".join(profile.transmitted)
    assert "[EMAIL]" in blob
    assert "[PHONE]" in blob
    assert RAW_EMAIL not in blob
    assert RAW_PHONE not in blob
    assert "all good" in blob


def test_harmless_column_keeps_its_own_samples() -> None:
    profile = mapper.profile_column("Product", _frame()["Product"])
    assert not profile.sensitive
    assert profile.transmitted == ["Widget", "Gadget", "Sprocket"]


def test_payload_contains_no_raw_pii() -> None:
    text = _payload_text(headers_only=False)
    for secret in FORBIDDEN:
        assert secret not in text, secret
    payload = json.loads(text)
    for column in payload["sources"]:
        if column["sensitive"]:
            assert "masked_samples" not in column
            patterns = " ".join(column.get("patterns", []))
            assert "[" not in patterns or "[EMAIL]" not in patterns
        else:
            assert "patterns" not in column


def test_headers_only_sends_no_samples() -> None:
    text = _payload_text(headers_only=True)
    payload = json.loads(text)
    assert payload["headers_only"] is True
    for column in payload["sources"] + payload["targets"]:
        assert "patterns" not in column
        assert "masked_samples" not in column
    assert "Widget" not in text
    assert "Gadget" not in text
    for secret in FORBIDDEN:
        assert secret not in text


def test_header_tokens_do_not_overmatch_ordinary_words() -> None:
    assert "name" in mapper.header_sensitivity_hits("Full Name")
    assert "name" in mapper.header_sensitivity_hits("fullname")
    assert mapper.header_sensitivity_hits("filename") == []
    assert mapper.header_sensitivity_hits("company") == []
    assert "email" in mapper.header_sensitivity_hits("E-mail Address")
    assert "account" in mapper.header_sensitivity_hits("Account No")
    assert mapper.header_sensitivity_hits("quantity") == []


def test_at_most_three_samples_are_transmitted() -> None:
    series = pd.Series(["alpha", "beta", "gamma", "delta", "epsilon"])
    profile = mapper.profile_column("Label", series)
    assert profile.transmitted == ["alpha", "beta", "gamma"]


def test_blank_values_are_skipped() -> None:
    series = pd.Series([None, "  ", "kept", pd.NA])
    profile = mapper.profile_column("Label", series)
    assert profile.transmitted == ["kept"]


@pytest.mark.parametrize(
    "header",
    [
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
    ],
)
def test_every_sensitive_token_is_detected(header: str) -> None:
    assert header in mapper.header_sensitivity_hits(header)
