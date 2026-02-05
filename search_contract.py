#!/usr/bin/env python3
"""
search_contract.py - Interactive keyword search over extracted contract JSON.

Usage:
    python search_contract.py <extracted_json>
    python search_contract.py contract_extracted.json
    python search_contract.py contract_extracted.json --top 10

Features:
    - Multi-keyword queries (e.g. "termination clauses")
    - Relevance ranking based on keyword frequency and density
    - Context previews around matched terms
    - Case-insensitive matching
    - Interactive REPL with history
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

CONTEXT_CHARS = 120  # characters of context shown around each match


def load_contract(json_path: str) -> dict:
    """Load the extracted contract JSON file.

    Args:
        json_path: Path to the JSON file produced by process_contract.py.

    Returns:
        Parsed contract data dict.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_query(query: str) -> list[str]:
    """Split a query into individual search terms, lowercased.

    Args:
        query: Raw user query string.

    Returns:
        List of lowercase keyword strings.
    """
    return [t for t in query.lower().split() if t]


def score_page(page_text: str, keywords: list[str]) -> tuple[int, float]:
    """Score a page's relevance to the given keywords.

    Scoring considers:
      - Total keyword hits (primary sort)
      - Keyword density: hits / page_length (tiebreaker)

    Args:
        page_text: Full text of the page (lowercased externally).
        keywords: List of lowercase keywords.

    Returns:
        Tuple of (total_hits, density) for sorting.
    """
    total_hits = 0
    for kw in keywords:
        total_hits += page_text.count(kw)

    length = max(len(page_text), 1)
    density = total_hits / length
    return total_hits, density


def extract_snippets(page_text: str, keywords: list[str], max_snippets: int = 3) -> list[str]:
    """Pull short context snippets around keyword matches.

    Args:
        page_text: Original (not lowered) page text.
        keywords: Lowercase keywords to highlight.
        max_snippets: Maximum number of snippets to return.

    Returns:
        List of snippet strings with matched terms in [brackets].
    """
    text_lower = page_text.lower()
    seen_positions: set[int] = set()
    snippets: list[str] = []

    for kw in keywords:
        start = 0
        while start < len(text_lower):
            idx = text_lower.find(kw, start)
            if idx == -1:
                break

            # Avoid overlapping snippets
            if any(abs(idx - pos) < CONTEXT_CHARS for pos in seen_positions):
                start = idx + len(kw)
                continue

            seen_positions.add(idx)
            left = max(0, idx - CONTEXT_CHARS // 2)
            right = min(len(page_text), idx + len(kw) + CONTEXT_CHARS // 2)

            snippet = page_text[left:right]
            # Clean up whitespace
            snippet = " ".join(snippet.split())
            # Bracket the matched term for visibility
            snippet = re.sub(
                re.escape(kw),
                lambda m: f"[{m.group(0)}]",
                snippet,
                flags=re.IGNORECASE,
            )

            if left > 0:
                snippet = "..." + snippet
            if right < len(page_text):
                snippet = snippet + "..."

            snippets.append(snippet)
            start = idx + len(kw)

            if len(snippets) >= max_snippets:
                return snippets

    return snippets


def search(contract_data: dict, query: str, top_n: int = 5) -> list[dict]:
    """Search the contract for pages matching the query.

    Args:
        contract_data: Loaded contract JSON dict.
        query: Raw user query.
        top_n: Max results to return.

    Returns:
        List of result dicts sorted by relevance, each containing:
            page_number, hits, density, snippets
    """
    keywords = tokenize_query(query)
    if not keywords:
        return []

    results = []
    for page in contract_data["pages"]:
        text = page["text"]
        if not text:
            continue

        text_lower = text.lower()
        hits, density = score_page(text_lower, keywords)

        if hits == 0:
            continue

        snippets = extract_snippets(text, keywords)
        results.append({
            "page_number": page["page_number"],
            "hits": hits,
            "density": round(density, 6),
            "snippets": snippets,
        })

    results.sort(key=lambda r: (r["hits"], r["density"]), reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

SEPARATOR = "-" * 60


def print_results(results: list[dict], query: str, total_pages: int):
    """Pretty-print search results to the terminal.

    Args:
        results: List of result dicts from search().
        query: The original query string.
        total_pages: Total pages in the contract (for context).
    """
    print(f'\n{SEPARATOR}')
    print(f'  Results for: "{query}"')
    print(f'  Matched {len(results)} of {total_pages} pages')
    print(SEPARATOR)

    if not results:
        print("  No matches found. Try different keywords.\n")
        return

    for rank, result in enumerate(results, start=1):
        page = result["page_number"]
        hits = result["hits"]
        print(f"\n  #{rank}  Page {page}  ({hits} hit{'s' if hits != 1 else ''})")
        print(f"  {'~' * 40}")
        for snippet in result["snippets"]:
            wrapped = textwrap.fill(snippet, width=72, initial_indent="    ", subsequent_indent="    ")
            print(wrapped)
    print()


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_search(json_path: str, top_n: int):
    """Run an interactive search REPL.

    Args:
        json_path: Path to the extracted contract JSON.
        top_n: Default number of top results to show.
    """
    contract_data = load_contract(json_path)
    source = contract_data.get("source_file", "unknown")
    total = contract_data.get("total_pages", 0)

    print(f"\nContract Search")
    print(f"  Source : {source}")
    print(f"  Pages : {total}")
    print(f"  Top-N : {top_n}")
    print(f"\nType a query and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        results = search(contract_data, query, top_n=top_n)
        print_results(results, query, total)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive keyword search over an extracted contract JSON."
    )
    parser.add_argument("json_file", help="Path to the extracted contract JSON")
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top results to display (default: 5)",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    interactive_search(str(path), top_n=args.top)


if __name__ == "__main__":
    main()
