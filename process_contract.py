#!/usr/bin/env python3
"""
Process Contract - Extract and save text from PDF contracts
Extracts text from PDF once and saves to JSON for faster searching
"""

import PyPDF2
import json
import sys
import os
from pathlib import Path


def extract_contract_text(pdf_path):
    """
    Extract text from PDF with page numbers

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with page number and text
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_text = []

            total_pages = len(pdf_reader.pages)
            print(f"Processing {total_pages} pages...")

            for page_num in range(total_pages):
                text = pdf_reader.pages[page_num].extract_text()
                pages_text.append({
                    'page': page_num + 1,
                    'text': text
                })

                # Progress indicator
                if (page_num + 1) % 10 == 0 or page_num == total_pages - 1:
                    print(f"  Processed {page_num + 1}/{total_pages} pages")

            return pages_text
    except FileNotFoundError:
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def save_extracted_text(pages_text, output_path):
    """
    Save extracted text to JSON file

    Args:
        pages_text: List of page dictionaries
        output_path: Path to output JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_text, f, indent=2, ensure_ascii=False)
        print(f"\nExtracted text saved to: {output_path}")

        # Calculate file size
        file_size = os.path.getsize(output_path)
        size_kb = file_size / 1024
        print(f"File size: {size_kb:.1f} KB")

        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 process_contract.py <path_to_pdf>")
        print("\nExample:")
        print("  python3 process_contract.py contract.pdf")
        print("  python3 process_contract.py DMS-2122-027CGroupDentalContract(Ameritas).pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Generate output filename
    pdf_name = Path(pdf_path).stem
    output_path = f"{pdf_name}_extracted.json"

    print("=" * 70)
    print("Contract Processing Tool")
    print("=" * 70)
    print(f"Input PDF: {pdf_path}")
    print(f"Output file: {output_path}")
    print()

    # Extract text
    pages_text = extract_contract_text(pdf_path)

    if not pages_text:
        print("Failed to extract text from PDF")
        sys.exit(1)

    # Save to JSON
    success = save_extracted_text(pages_text, output_path)

    if success:
        print("\n" + "=" * 70)
        print("Processing complete!")
        print("=" * 70)
        print(f"\nYou can now search this contract using:")
        print(f"  python3 search_contract.py {output_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
