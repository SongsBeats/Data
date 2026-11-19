# Header Mapper

A small page that renames the columns of a source sheet so they line up with a target sheet's header row.

Local matching runs first. Gemini reviews only the headers that are still empty, using the masked payload shown on the page. The API key stays on the server.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Exact and fuzzy matches do not need a key. To review empty headers, set `GEMINI_API_KEY` in the environment. Optional: `GEMINI_MODEL` (default `gemini-2.5-flash`).

Sample files: `samples/source_orders.xlsx` (sheet Orders) and `samples/target_template.csv`. The target file has a data row that the app must not read.

## Deploy on Vercel

The app is a FastAPI function plus the static page in `public/`. Vercel serves `public/` from the CDN and `app.py` as one Python function. Uploads stay under 4 MB because Vercel request bodies are limited to 4.5 MB.

Set `GEMINI_API_KEY` in the project environment. The page does not ask for a key or a model name. Gemini runs only when some headers are still empty.

## What the page does

1. Choose the source sheet and the target sheet. Excel files with more than one sheet ask you to pick one. The target upload reads the header row only.
2. Match headers. Each target header has a source dropdown, a confidence label, a reason, and a status: Matched, Needs review, or Unmapped.
3. Download CSV or Excel in the target's column order. Unmapped targets are blank.

## Matching rules

- Headers are compared after lowercasing and removing punctuation and spaces. `E-mail` and `email` are an exact match.
- Remaining headers are scored with RapidFuzz (or `difflib` if RapidFuzz is not installed). The default threshold is 0.90.
- A source column is used at most once. If two candidates are too close, nothing is assigned.
- Scores under 0.60 are never assigned. The model is told to return `null` when it is unsure.

## Duplicates

Duplicate headers become `name`, `name__2`, `name__3` on screen. The download uses the original text again.

Columns with the same values are flagged. The source file line and **Remove fully duplicate rows** show how many rows are duplicates. The result repeats that count, whether those rows are kept or removed.

If you pick a source column that is already mapped, the page refuses the change and tells you to clear the other row first.

## Privacy

Uploaded files are not written to disk and are not logged.

A column is treated as sensitive when its header contains a token such as `email`, `phone`, `name`, `address`, `aadhaar`, `pan`, `ssn`, `salary`, or `password`, or when at least 30% of the sampled values match a PII pattern. Profiling looks at the first 200 rows.

- Sensitive columns send shape patterns only: `A` upper case, `a` lower case, `9` digit.
- Other columns send at most three samples, with inline PII replaced by tags such as `[EMAIL]` and `[PHONE]`.
- **Headers only** sends names and types, and no samples.

## Tests

```powershell
python -m pytest
```
