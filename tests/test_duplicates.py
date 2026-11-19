"""Duplicate headers, identical columns, one-to-one mapping, and row removal."""

from __future__ import annotations

import pandas as pd
import pytest

import mapper


def test_duplicate_headers_are_suffixed_and_listed() -> None:
    working, originals, duplicates = mapper.uniquify_headers(
        ["name", "age", "name", "name", "city"]
    )
    assert working == ["name", "age", "name__2", "name__3", "city"]
    assert originals == {
        "name": "name",
        "age": "age",
        "name__2": "name",
        "name__3": "name",
        "city": "city",
    }
    assert duplicates == ["name"]


def test_blank_headers_get_a_stable_name() -> None:
    working, originals, duplicates = mapper.uniquify_headers(["", "  ", "id"])
    assert working[0] == "column"
    assert working[1] == "column__2"
    assert originals[working[1]] == "column"
    assert duplicates == ["column"]


def test_load_table_warns_and_keeps_both_source_columns() -> None:
    loaded = mapper.load_table(b"name,name\nAda,Grace\n", "people.csv")
    assert list(loaded.frame.columns) == ["name", "name__2"]
    assert loaded.duplicate_names == ["name"]
    assert loaded.originals["name__2"] == "name"
    assert loaded.frame.iloc[0]["name__2"] == "Grace"


def test_identical_content_is_flagged() -> None:
    frame = pd.DataFrame(
        {
            "a": [1, 2, None],
            "b": [1, 2, None],
            "c": [1, 2, 3],
        }
    )
    assert mapper.find_identical_columns(frame) == [["a", "b"]]


def test_one_to_one_manual_assignment_asks_to_reassign() -> None:
    mappings = [
        mapper.Mapping("email", "Email", 1.0, "Exact match.", "Matched", "exact"),
        mapper.Mapping("mobile", None, 0.0, "No local match.", "Unmapped", "none"),
    ]
    error = mapper.apply_manual_choice(mappings, "mobile", "Email")
    assert error is not None
    assert "Email" in error
    assert mappings[1].source is None

    assert mapper.apply_manual_choice(mappings, "mobile", "Phone") is None
    assert mappings[1].source == "Phone"
    assert mappings[1].status == "Matched"
    assert mappings[1].origin == "manual"
    assert mappings[1].confidence == 1.0

    assert mapper.apply_manual_choice(mappings, "mobile", mapper.CLEAR) is None
    assert mappings[1].source is None
    assert mappings[1].status == "Unmapped"


def test_exact_match_uses_each_source_once() -> None:
    rows = mapper.local_match(["Email"], ["E-mail", "email"])
    sources = [row.source for row in rows]
    assert sources.count("Email") == 1
    assert sources.count(None) == 1


def test_duplicate_rows_are_counted_and_removed_in_target_order() -> None:
    source = pd.DataFrame(
        {
            "Price": [1, 1, 2],
            "Tax": [3, 3, 4],
            "Extra": [9, 9, 9],
        }
    )
    assert mapper.duplicate_row_count(source) == 1
    mappings = [
        mapper.Mapping("Amount", "Price", 1.0, "manual", "Matched", "manual"),
        mapper.Mapping("Amount__2", "Tax", 1.0, "manual", "Matched", "manual"),
        mapper.Mapping("Coupon", None, 0.0, "No match.", "Unmapped", "none"),
    ]
    output, summary = mapper.convert_frame(
        source,
        ["Amount", "Amount__2", "Coupon"],
        {"Amount": "Amount", "Amount__2": "Amount", "Coupon": "Coupon"},
        mappings,
        drop_duplicate_rows=True,
    )
    assert list(output.columns) == ["Amount", "Amount", "Coupon"]
    assert summary.columns_mapped == 2
    assert summary.unmapped_targets == 1
    assert summary.rows_in == 3
    assert summary.rows_removed == 1
    assert summary.duplicate_rows == 1
    assert summary.rows_kept == 2
    assert output.iloc[0, 0] == 1
    assert output.iloc[0, 1] == 3
    assert pd.isna(output.iloc[0, 2])
    assert len(output) == 2


def test_keeping_duplicate_rows_is_the_default() -> None:
    source = pd.DataFrame({"Price": [1, 1], "Tax": [3, 3]})
    mappings = [
        mapper.Mapping("Amount", "Price", 1.0, "", "Matched", "exact"),
        mapper.Mapping("Tax Out", "Tax", 1.0, "", "Matched", "exact"),
    ]
    output, summary = mapper.convert_frame(
        source,
        ["Amount", "Tax Out"],
        {"Amount": "Amount", "Tax Out": "Tax Out"},
        mappings,
        drop_duplicate_rows=False,
    )
    assert summary.rows_removed == 0
    assert summary.duplicate_rows == 1
    assert summary.rows_kept == 2
    assert len(output) == 2


def test_output_restores_original_target_names_not_suffixes() -> None:
    source = mapper.load_table(b"cost,cost\n4,5\n", "costs.csv").frame
    mappings = [
        mapper.Mapping("Fee", "cost", 1.0, "", "Matched", "exact"),
        mapper.Mapping("Fee__2", "cost__2", 1.0, "", "Matched", "exact"),
    ]
    output, _summary = mapper.convert_frame(
        source,
        ["Fee", "Fee__2"],
        {"Fee": "Fee", "Fee__2": "Fee"},
        mappings,
        drop_duplicate_rows=False,
    )
    assert list(output.columns) == ["Fee", "Fee"]
    assert list(output.iloc[0]) == [4, 5]
