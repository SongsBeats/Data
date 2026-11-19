"""File loading: encodings, header-only reads, empty files, and size limits."""

from __future__ import annotations

import io

import pandas as pd
import pytest

import mapper


def test_empty_file_is_rejected() -> None:
    with pytest.raises(mapper.EmptyFileError):
        mapper.load_table(b"", "empty.csv")
    with pytest.raises(mapper.EmptyFileError):
        mapper.load_table(b"   \n  \n", "blank.csv")


def test_missing_header_row_is_rejected() -> None:
    with pytest.raises(mapper.MissingHeaderError):
        mapper.load_table(b",\n1,2\n", "no-header.csv")


def test_utf8_sig_and_cp1252_are_read() -> None:
    utf8 = mapper.load_table("city,note\nPune,café\n".encode("utf-8-sig"), "a.csv")
    assert utf8.frame.iloc[0]["note"] == "café"

    cp1252 = mapper.load_table(b"title\nHello\x93World\n", "b.csv")
    assert "World" in cp1252.frame.iloc[0]["title"]


def test_oversized_file_is_rejected_before_parsing() -> None:
    with pytest.raises(mapper.FileTooLargeError):
        mapper.load_table(b"a,b\n1,2\n", "big.csv", max_bytes=4)


def test_target_read_never_includes_data_rows() -> None:
    loaded = mapper.load_table(
        b"email,name\nsecret@secret.com,SECRETNAME\n",
        "target.csv",
        header_only=True,
    )
    assert list(loaded.frame.columns) == ["email", "name"]
    assert len(loaded.frame) == 0
    assert loaded.row_count == 0
    assert "secret@secret.com" not in loaded.frame.to_csv()
    assert "SECRETNAME" not in loaded.frame.to_csv()


def test_excel_sheet_selection_and_header_only() -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": ["secret-data"]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"Keep": [1], "Drop": [2]}).to_excel(writer, sheet_name="Second", index=False)
    data = buffer.getvalue()

    assert mapper.list_sheets(data, "book.xlsx") == ["First", "Second"]

    headers = mapper.load_table(data, "book.xlsx", sheet="First", header_only=True)
    assert list(headers.frame.columns) == ["a"]
    assert len(headers.frame) == 0
    assert "secret-data" not in headers.frame.to_csv()

    body = mapper.load_table(data, "book.xlsx", sheet="Second")
    assert list(body.frame.columns) == ["Keep", "Drop"]
    assert body.row_count == 1
    assert body.frame.iloc[0]["Keep"] == 1


def test_csv_has_no_sheets() -> None:
    assert mapper.list_sheets(b"a,b\n1,2\n", "plain.csv") == []
