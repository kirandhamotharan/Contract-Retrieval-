# Contract Retrieval Tool

A tool for searching contract PDFs using natural language queries. Find termination clauses, renewal terms, payment details, and more using keyword-based search.

## Features

- Extract text from PDF contracts once and save for fast searching
- Smart keyword expansion (e.g., "when does contract end" finds termination clauses)
- Interactive search mode or single-query mode
- Context preview around matches
- Page number references

## Installation

### Requirements
- Python 3.7+
- PyPDF2

### Setup

```bash
# Install dependencies
pip install PyPDF2

# Or if using pip3
pip3 install PyPDF2
```

## Usage

### Step 1: Process your PDF contract

First, extract text from your PDF contract:

```bash
python3 process_contract.py your_contract.pdf
```

This creates a JSON file (e.g., `your_contract_extracted.json`) with the extracted text.

**Example:**
```bash
python3 process_contract.py DMS-2122-027CGroupDentalContract(Ameritas).pdf
# Creates: DMS-2122-027CGroupDentalContract(Ameritas)_extracted.json
```

### Step 2: Search the contract

#### Interactive Mode

Run interactive search to ask multiple questions:

```bash
python3 search_contract.py your_contract_extracted.json
```

Then type your queries:
```
Search: when does this contract end
Search: what are the payment terms
Search: termination clause
Search: quit
```

#### Single Query Mode

Search with a single query:

```bash
python3 search_contract.py your_contract_extracted.json "termination clause"
python3 search_contract.py your_contract_extracted.json "when does contract end"
python3 search_contract.py your_contract_extracted.json "payment terms"
```

## Example Queries

The tool understands common contract queries:

- **Termination:** "when does this contract end", "how to cancel", "termination clause"
- **Renewal:** "renewal terms", "automatic renewal", "how to renew"
- **Payment:** "payment terms", "fees", "billing", "cost"
- **Term:** "contract duration", "term length", "effective date"
- **Liability:** "liability", "indemnity", "damages"
- **Confidentiality:** "confidential information", "NDA terms"
- **Insurance:** "insurance requirements", "coverage"
- **Notice:** "notice requirements", "how to notify"

## How It Works

### Smart Keyword Expansion

When you search for "when does this contract end", the tool automatically expands this to search for:
- termination
- terminate
- cancel
- cancellation
- end contract

This increases the chance of finding relevant sections.

### Relevance Ranking

Results are ranked by:
1. Number of keyword matches (pages with more matches appear first)
2. Context preview shows text around the first match

## Project Structure

```
contract-retrieval/
├── process_contract.py          # Extract text from PDF
├── search_contract.py           # Search extracted text
├── extract_pdf.py               # Legacy: simple PDF extraction
├── contract_search.py           # Legacy: original search implementation
├── README.md                    # This file
├── your_contract.pdf            # Your PDF contract
└── your_contract_extracted.json # Extracted text (generated)
```

## Future Enhancements (Option B)

Currently exploring semantic search with:
- sentence-transformers for better understanding of queries
- Vector embeddings for semantic similarity
- More natural language understanding

This requires additional dependencies and is planned for a future version.

## Troubleshooting

### "File not found" error
Make sure you run `process_contract.py` first to extract the PDF text.

### No results found
- Try using different keywords
- Use more general terms (e.g., "payment" instead of "monthly payment")
- Check the PDF extracted correctly by opening the JSON file

### PyPDF2 import error
```bash
pip3 install PyPDF2
```

## Development Notes

### Current Status (Option A - Keyword Search)
- ✅ PDF text extraction
- ✅ Keyword-based search
- ✅ Smart query expansion
- ✅ Interactive search mode
- ✅ Results ranking

### Planned (Option B - Semantic Search)
- ⏳ sentence-transformers integration
- ⏳ Vector embeddings
- ⏳ Semantic similarity search
- ⏳ Question answering

## License

This is a personal project for contract analysis.

## Contributing

This is currently a personal tool. Feel free to fork and adapt for your needs.

---

**Quick Start:**
```bash
# 1. Extract PDF
python3 process_contract.py contract.pdf

# 2. Search
python3 search_contract.py contract_extracted.json

# 3. Ask questions
Search: when does this contract end?
```
