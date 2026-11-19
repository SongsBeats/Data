"""API checks: target data never comes back, and the model payload has no raw PII."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
from app import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def _source() -> tuple[str, bytes, str]:
    path = ROOT / "samples" / "source_orders.xlsx"
    return ("source_orders.xlsx", path.read_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _target() -> tuple[str, bytes, str]:
    path = ROOT / "samples" / "target_template.csv"
    return ("target_template.csv", path.read_bytes(), "text/csv")


def test_target_response_contains_headers_only() -> None:
    response = client.post("/api/target", files={"file": _target()})
    assert response.status_code == 200
    body = response.text
    assert "customer_name" in body
    assert "secret@secret.com" not in body
    assert "DO_NOT_READ" not in body
    assert "SAVE10" not in body


def test_match_payload_has_shapes_not_raw_pii() -> None:
    response = client.post(
        "/api/match",
        files={"source": _source(), "target": _target()},
        data={"source_sheet": "Orders", "threshold": "0.90"},
    )
    assert response.status_code == 200
    body = response.text
    assert "priya.shah@example.com" not in body
    assert "9876543210" not in body
    assert "12 River Road" not in body
    assert "secret@secret.com" not in body
    payload = response.json()
    names = {item["header"]: item for item in payload["source_profiles"]}
    assert names["Phone Number"]["patterns"]
    assert "masked_samples" not in names["Phone Number"]
    email = next(row for row in payload["mappings"] if row["target"] == "email")
    assert email["source"] == "E-mail"
    assert email["status"] == "Matched"


def test_convert_drops_one_duplicate_row_and_keeps_target_order() -> None:
    matched = client.post(
        "/api/match",
        files={"source": _source(), "target": _target()},
        data={"source_sheet": "Orders"},
    ).json()
    response = client.post(
        "/api/convert",
        files={"source": _source(), "target": _target()},
        data={
            "source_sheet": "Orders",
            "mappings": __import__("json").dumps(matched["mappings"]),
            "drop_duplicates": "1",
            "preview": "1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows_in"] == 4
    assert body["rows_removed"] == 1
    assert body["duplicate_rows"] == 1
    assert body["rows_kept"] == 3
    assert body["preview"]["columns"][0] == "customer_name"
    assert "secret@secret.com" not in response.text


def test_duplicate_source_assignment_is_rejected() -> None:
    matched = client.post(
        "/api/match",
        files={"source": _source(), "target": _target()},
        data={"source_sheet": "Orders"},
    ).json()
    for row in matched["mappings"]:
        if row["target"] == "mobile":
            row["source"] = "E-mail"
    response = client.post(
        "/api/convert",
        files={"source": _source(), "target": _target()},
        data={
            "source_sheet": "Orders",
            "mappings": __import__("json").dumps(matched["mappings"]),
            "preview": "1",
        },
    )
    assert response.status_code == 400
    assert "already" in response.json()["error"] or "more than once" in response.json()["error"]


def test_suggest_without_gemini_key_leaves_local_mappings(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    matched = client.post(
        "/api/match",
        files={"source": _source(), "target": _target()},
        data={"source_sheet": "Orders"},
    ).json()
    response = client.post(
        "/api/suggest",
        files={"source": _source(), "target": _target()},
        data={
            "source_sheet": "Orders",
            "mappings": json.dumps(matched["mappings"]),
            "headers_only": "1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["checked"] is False
    email = next(row for row in body["mappings"] if row["target"] == "email")
    assert email["source"] == "E-mail"
    customer = next(row for row in body["mappings"] if row["target"] == "customer_name")
    assert customer["source"] is None


def test_suggest_sends_shapes_and_merges_gemini_reply(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen: dict[str, str] = {}

    def fake_gemini(api_key: str, prompt: str) -> str:
        seen["key"] = api_key
        seen["prompt"] = prompt
        return json.dumps(
            [
                {
                    "target": "customer_name",
                    "source": "Cust Name",
                    "confidence": 0.8,
                    "reason": "Both name a customer.",
                }
            ]
        )

    monkeypatch.setattr(app_module, "_call_gemini", fake_gemini)
    matched = client.post(
        "/api/match",
        files={"source": _source(), "target": _target()},
        data={"source_sheet": "Orders"},
    ).json()
    response = client.post(
        "/api/suggest",
        files={"source": _source(), "target": _target()},
        data={
            "source_sheet": "Orders",
            "mappings": json.dumps(matched["mappings"]),
            "headers_only": "0",
        },
    )
    assert response.status_code == 200
    assert seen["key"] == "test-key"
    assert "priya.shah@example.com" not in seen["prompt"]
    assert "9876543210" not in seen["prompt"]
    assert "Aaaaa Aaaa" in seen["prompt"]
    body = response.json()
    assert body["checked"] is True
    customer = next(row for row in body["mappings"] if row["target"] == "customer_name")
    assert customer["source"] == "Cust Name"
    assert customer["status"] == "Needs review"
    email = next(row for row in body["mappings"] if row["target"] == "email")
    assert email["source"] == "E-mail"
