#!/usr/bin/env python3
"""
Contract Search - Interactive search tool for contract documents
Searches extracted contract text using keyword matching
"""

import json
import sys
import re
from pathlib import Path


# Common contract queries mapped to keywords
QUERY_TEMPLATES = {
    'termination': ['termination', 'terminate', 'cancel', 'cancellation', 'end contract'],
    'renewal': ['renewal', 'renew', 'extend', 'extension', 'automatic renewal'],
    'payment': ['payment', 'fee', 'cost', 'price', 'billing', 'invoice'],
    'liability': ['liability', 'indemnity', 'indemnification', 'damages'],
    'term': ['term', 'duration', 'period', 'effective date', 'expiration'],
    'confidentiality': ['confidential', 'confidentiality', 'proprietary', 'nda'],
    'insurance': ['insurance', 'coverage', 'policy', 'insured'],
    'notice': ['notice', 'notification', 'notify', 'written notice'],
}


def load_extracted_text(json_path):
    """
    Load extracted contract text from JSON file

    Args:
        json_path: Path to JSON file with extracted text

    Returns:
        List of page dictionaries
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        print("\nDid you run process_contract.py first?")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def expand_query(query):
    """
    Expand query with related terms from templates

    Args:
        query: User's search query

    Returns:
        List of keywords to search for
    """
    query_lower = query.lower()
    keywords = set(query_lower.split())

    # Check if query matches any template
    for category, terms in QUERY_TEMPLATES.items():
        for term in terms:
            if term in query_lower:
                # Add all related terms from this category
                keywords.update(terms)
                break

    return list(keywords)


def search_contract(pages_text, query, use_smart_search=True):
    """
    Search for keywords in contract

    Args:
        pages_text: List of page dictionaries
        query: Search query string
        use_smart_search: Whether to expand query with related terms

    Returns:
        List of matching pages with relevance scores
    """
    if use_smart_search:
        keywords = expand_query(query)
    else:
        keywords = query.lower().split()

    results = []

    for page_data in pages_text:
        text = page_data['text'].lower()

        # Count how many keywords appear
        matches = sum(1 for keyword in keywords if keyword in text)

        if matches > 0:
            # Get context around keywords (preview)
            preview = get_preview(page_data['text'], keywords)
            results.append({
                'page': page_data['page'],
                'matches': matches,
                'preview': preview,
                'full_text': page_data['text']
            })

    # Sort by number of matches (most relevant first)
    results.sort(key=lambda x: x['matches'], reverse=True)
    return results


def get_preview(text, keywords, context_chars=200):
    """
    Get text snippet around first keyword match

    Args:
        text: Full text to search
        keywords: List of keywords
        context_chars: Number of characters to show around match

    Returns:
        Preview string with context
    """
    text_lower = text.lower()

    # Find first keyword match
    for keyword in keywords:
        pos = text_lower.find(keyword)
        if pos != -1:
            start = max(0, pos - context_chars)
            end = min(len(text), pos + context_chars)
            snippet = text[start:end].strip()

            # Add ellipsis if truncated
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."

            return snippet

    # Fallback: return first part of text
    return text[:400].strip() + "..."


def display_results(results, query, show_full=False, max_results=5):
    """
    Display search results

    Args:
        results: List of search results
        query: Original search query
        show_full: Whether to show full page text
        max_results: Maximum number of results to display
    """
    if not results:
        print("\nNo results found.")
        print("\nTry using different keywords or related terms.")
        return

    print(f"\nFound {len(results)} pages with matches")
    print("=" * 70)

    # Show top results
    for i, result in enumerate(results[:max_results], 1):
        print(f"\n[Result {i}] Page {result['page']} - {result['matches']} keyword matches")
        print("-" * 70)
        print(f"Preview: {result['preview']}")

        if show_full:
            print("\nFull page text:")
            print(result['full_text'])
            print("-" * 70)

    if len(results) > max_results:
        print(f"\n... and {len(results) - max_results} more results")
        print(f"Use --all to see all results or --full to see complete text")


def interactive_search(json_path):
    """
    Interactive search mode

    Args:
        json_path: Path to extracted contract JSON
    """
    print("=" * 70)
    print("Contract Search Tool - Interactive Mode")
    print("=" * 70)

    # Load contract
    print(f"\nLoading contract from: {json_path}")
    pages_text = load_extracted_text(json_path)

    if not pages_text:
        return

    print(f"Loaded {len(pages_text)} pages")
    print("\nAvailable smart queries:")
    for category in sorted(QUERY_TEMPLATES.keys()):
        print(f"  - {category}")

    print("\nEnter your search queries (or 'quit' to exit)")
    print("=" * 70)

    while True:
        try:
            query = input("\nSearch: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Search
            results = search_contract(pages_text, query)
            display_results(results, query)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function"""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 search_contract.py <extracted_json> [query]")
        print("\nExamples:")
        print("  Interactive mode:")
        print("    python3 search_contract.py contract_extracted.json")
        print("\n  Single query:")
        print("    python3 search_contract.py contract_extracted.json 'termination clause'")
        print("    python3 search_contract.py contract_extracted.json 'when does this contract end'")
        sys.exit(1)

    json_path = sys.argv[1]

    # Check if running in interactive or single-query mode
    if len(sys.argv) == 2:
        # Interactive mode
        interactive_search(json_path)
    else:
        # Single query mode
        query = ' '.join(sys.argv[2:])

        print("=" * 70)
        print("Contract Search Tool")
        print("=" * 70)

        # Load contract
        pages_text = load_extracted_text(json_path)
        if not pages_text:
            sys.exit(1)

        print(f"Loaded {len(pages_text)} pages")
        print(f"Searching for: '{query}'")

        # Search
        results = search_contract(pages_text, query)
        display_results(results, query, max_results=10)

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
