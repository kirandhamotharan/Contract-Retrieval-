# Contract Search Pipeline

Extract text from contract PDFs and search them by keyword from the command line.

## Requirements

- Python 3.10+
- PyPDF2

```bash
pip install PyPDF2
```

## Quick Start

### 1. Extract text from a PDF

```bash
python process_contract.py "DMS-2122-027CGroupDentalContract(Ameritas).pdf"
```

This produces a JSON file (e.g. `DMS-2122-027CGroupDentalContract(Ameritas)_extracted.json`) containing the text of every page.

You can specify a custom output path:

```bash
python process_contract.py contract.pdf -o data/contract.json
```

### 2. Search the extracted contract

```bash
python search_contract.py DMS-2122-027CGroupDentalContract\(Ameritas\)_extracted.json
```

This opens an interactive prompt:

```
Contract Search
  Source : DMS-2122-027CGroupDentalContract(Ameritas).pdf
  Pages : 42
  Top-N : 5

Type a query and press Enter. Type 'quit' to exit.

search> termination clauses
```

Results show matched pages ranked by relevance with context snippets:

```
------------------------------------------------------------
  Results for: "termination clauses"
  Matched 3 of 42 pages
------------------------------------------------------------

  #1  Page 12  (7 hits)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...the Group may elect [termination] of this contract
    upon thirty (30) days written notice. Such [clauses]
    apply to...
```

Show more results with `--top`:

```bash
python search_contract.py contract_extracted.json --top 10
```

## Example Queries

| Query | What it finds |
|---|---|
| `termination clauses` | Cancellation and termination provisions |
| `payment terms` | Payment schedules, due dates, billing |
| `renewal conditions` | Auto-renewal, renewal notice periods |
| `eligibility` | Who is covered, enrollment rules |
| `waiting period` | Service waiting periods, effective dates |
| `appeal` | Grievance and appeals processes |

## JSON Format

The extracted JSON follows this structure:

```json
{
  "source_file": "contract.pdf",
  "total_pages": 42,
  "extracted_at": "2026-02-05T12:00:00",
  "pages": [
    {"page_number": 1, "text": "..."},
    {"page_number": 2, "text": "..."}
  ]
}
```

## Project Structure

```
process_contract.py   - PDF text extraction -> JSON
search_contract.py    - Interactive keyword search over JSON
README.md             - This file
```
