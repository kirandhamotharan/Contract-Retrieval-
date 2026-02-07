#!/usr/bin/env python3
"""
semantic_search.py - Semantic search over an extracted contract JSON.

Uses sentence-transformers to embed contract text into vectors, then
finds the most semantically similar passages to a natural-language query.
Returns citations with page numbers and context -- no AI-generated answers.

Usage:
    python semantic_search.py <extracted_json>
    python semantic_search.py contract_extracted.json
    python semantic_search.py contract_extracted.json --top 5 --model all-MiniLM-L6-v2

First run builds an embedding cache (saved alongside the JSON).
Subsequent runs load from cache and start instantly.

Requirements:
    pip install sentence-transformers
"""

import argparse
import json
import os
import pickle
import sys
import textwrap
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers is required.")
    print("Install it with: pip install sentence-transformers")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
SNIPPET_WIDTH = 200     # characters of context shown per result
SEPARATOR = "-" * 60


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for finer-grained embeddings.

    Args:
        text: Full page text.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks. Returns the original text as a single
        chunk if it is shorter than chunk_size.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_chunks(contract_data: dict) -> tuple[list[str], list[int]]:
    """Build chunk list from all pages, tracking which page each chunk belongs to.

    Args:
        contract_data: Loaded contract JSON dict.

    Returns:
        Tuple of (chunks, page_numbers) where page_numbers[i] is the
        source page for chunks[i].
    """
    chunks = []
    page_numbers = []

    for page in contract_data["pages"]:
        text = page["text"]
        if not text or not text.strip():
            continue

        page_chunks = chunk_text(text)
        chunks.extend(page_chunks)
        page_numbers.extend([page["page_number"]] * len(page_chunks))

    return chunks, page_numbers


# ---------------------------------------------------------------------------
# Embedding + caching
# ---------------------------------------------------------------------------

def cache_path_for(json_path: str, model_name: str) -> Path:
    """Determine the cache file path for a given JSON + model combination.

    Args:
        json_path: Path to the source JSON file.
        model_name: Name of the sentence-transformers model.

    Returns:
        Path to the .pkl cache file.
    """
    p = Path(json_path)
    safe_model = model_name.replace("/", "_")
    return p.with_suffix(f".{safe_model}.cache.pkl")


def load_or_build_embeddings(
    json_path: str,
    contract_data: dict,
    model_name: str,
) -> tuple[np.ndarray, list[str], list[int], SentenceTransformer]:
    """Load cached embeddings or build them from scratch.

    Args:
        json_path: Path to the extracted contract JSON.
        contract_data: Already-loaded contract data dict.
        model_name: sentence-transformers model name.

    Returns:
        Tuple of (embeddings, chunks, page_numbers, model).
    """
    cache_file = cache_path_for(json_path, model_name)

    # Try loading from cache
    if cache_file.exists():
        json_mtime = os.path.getmtime(json_path)
        cache_mtime = os.path.getmtime(cache_file)

        if cache_mtime >= json_mtime:
            print(f"  Loading cached embeddings from {cache_file.name}")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            model = SentenceTransformer(model_name)
            return (
                cached["embeddings"],
                cached["chunks"],
                cached["page_numbers"],
                model,
            )

    # Build from scratch
    print(f"  Building embeddings with model: {model_name}")
    chunks, page_numbers = build_chunks(contract_data)

    if not chunks:
        print("Error: No text found in the contract JSON.")
        sys.exit(1)

    print(f"  Chunked {contract_data['total_pages']} pages into {len(chunks)} segments")

    model = SentenceTransformer(model_name)
    print("  Encoding chunks (this may take a moment on first run)...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # Save cache
    with open(cache_file, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "chunks": chunks,
            "page_numbers": page_numbers,
        }, f)
    print(f"  Cached embeddings to {cache_file.name}")

    return embeddings, chunks, page_numbers, model


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[str],
    page_numbers: list[int],
    top_n: int = 5,
) -> list[dict]:
    """Find the most semantically similar chunks to the query.

    Uses cosine similarity between the query embedding and all chunk
    embeddings. Results are deduplicated by page -- only the best-scoring
    chunk per page is kept.

    Args:
        query: Natural-language search query.
        model: Loaded SentenceTransformer model.
        embeddings: Pre-computed chunk embeddings (N x D).
        chunks: List of text chunks parallel to embeddings.
        page_numbers: Page number for each chunk.
        top_n: Maximum number of results to return.

    Returns:
        List of result dicts sorted by descending similarity, each with:
            page_number, score, snippet
    """
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Cosine similarity (embeddings are already normalized by most models)
    scores = np.dot(embeddings, query_embedding.T).flatten()

    # Deduplicate: keep best chunk per page
    best_by_page: dict[int, tuple[float, str]] = {}
    for idx in range(len(scores)):
        page = page_numbers[idx]
        score = float(scores[idx])
        if page not in best_by_page or score > best_by_page[page][0]:
            best_by_page[page] = (score, chunks[idx])

    # Sort by score descending
    ranked = sorted(best_by_page.items(), key=lambda x: x[1][0], reverse=True)

    results = []
    for page, (score, chunk) in ranked[:top_n]:
        snippet = " ".join(chunk.split())
        if len(snippet) > SNIPPET_WIDTH:
            snippet = snippet[:SNIPPET_WIDTH].rsplit(" ", 1)[0] + "..."
        results.append({
            "page_number": page,
            "score": round(score, 4),
            "snippet": snippet,
        })

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results: list[dict], query: str, total_pages: int):
    """Pretty-print search results to the terminal.

    Args:
        results: List of result dicts from semantic_search().
        query: The original query string.
        total_pages: Total pages in the contract.
    """
    print(f"\n{SEPARATOR}")
    print(f'  Results for: "{query}"')
    print(f"  Showing top {len(results)} of {total_pages} pages")
    print(SEPARATOR)

    if not results:
        print("  No relevant matches found. Try rephrasing your query.\n")
        return

    for rank, result in enumerate(results, start=1):
        page = result["page_number"]
        score = result["score"]
        snippet = result["snippet"]

        print(f"\n  #{rank}  Page {page}  (similarity: {score:.4f})")
        print(f"  {'~' * 40}")
        wrapped = textwrap.fill(
            snippet, width=72,
            initial_indent="    ", subsequent_indent="    ",
        )
        print(wrapped)

    print()


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_search(json_path: str, model_name: str, top_n: int):
    """Run an interactive semantic search REPL.

    Args:
        json_path: Path to the extracted contract JSON.
        model_name: sentence-transformers model to use.
        top_n: Number of top results per query.
    """
    print(f"\nSemantic Contract Search")
    print(f"  Loading: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        contract_data = json.load(f)

    source = contract_data.get("source_file", "unknown")
    total = contract_data.get("total_pages", 0)

    embeddings, chunks, page_numbers, model = load_or_build_embeddings(
        json_path, contract_data, model_name,
    )

    print(f"\n  Source : {source}")
    print(f"  Pages  : {total}")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Model  : {model_name}")
    print(f"  Top-N  : {top_n}")
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

        results = semantic_search(
            query, model, embeddings, chunks, page_numbers, top_n=top_n,
        )
        print_results(results, query, total)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over an extracted contract JSON. "
        "Returns citations with page numbers -- no AI-generated answers.",
    )
    parser.add_argument(
        "json_file",
        help="Path to the extracted contract JSON (from process_contract.py)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top results to display (default: 5)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"sentence-transformers model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    interactive_search(str(path), model_name=args.model, top_n=args.top)


if __name__ == "__main__":
    main()
