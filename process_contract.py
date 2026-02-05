#!/usr/bin/env python3
"""
process_contract.py - Extract text from a contract PDF and save as structured JSON.

Usage:
    python process_contract.py <path_to_pdf>
    python process_contract.py contract.pdf
    python process_contract.py contract.pdf -o custom_output.json

Output JSON structure:
    {
        "source_file": "contract.pdf",
        "total_pages": 42,
        "extracted_at": "2026-02-05T12:00:00",
        "pages": [
            {"page_number": 1, "text": "..."},
            ...
        ]
    }
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("Error: PyPDF2 is required. Install it with: pip install PyPDF2")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with 'page_number' and 'text' keys.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        pages.append({"page_number": i, "text": text})

    return pages


def build_contract_json(pdf_path: str, pages: list[dict]) -> dict:
    """Build the structured JSON document from extracted pages.

    Args:
        pdf_path: Original PDF path (used for metadata).
        pages: List of extracted page dicts.

    Returns:
        Complete contract data dictionary.
    """
    return {
        "source_file": Path(pdf_path).name,
        "total_pages": len(pages),
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "pages": pages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a contract PDF and save as JSON."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <pdf_stem>_extracted.json)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    if not pdf_path.suffix.lower() == ".pdf":
        print(f"Warning: File does not have .pdf extension: {pdf_path}")

    output_path = Path(args.output) if args.output else pdf_path.with_name(
        f"{pdf_path.stem}_extracted.json"
    )

    print(f"Reading: {pdf_path}")
    pages = extract_text_from_pdf(str(pdf_path))

    empty_count = sum(1 for p in pages if not p["text"])
    if empty_count:
        print(f"Warning: {empty_count} page(s) had no extractable text.")

    contract_data = build_contract_json(str(pdf_path), pages)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contract_data, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(pages)} pages -> {output_path}")


if __name__ == "__main__":
    main()
