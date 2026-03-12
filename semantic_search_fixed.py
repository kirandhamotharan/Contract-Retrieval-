#!/usr/bin/env python3
"""
Semantic search over contract JSON using sentence-transformers embeddings.
Uses sentence-aware chunking to preserve clause boundaries for accurate retrieval.
"""

import sys
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"Error: {e}")
    print("Install required packages: pip install sentence-transformers rank_bm25")
    sys.exit(1)

from semantic_chunking import chunk_legal_document


def extract_section_number(text: str) -> Optional[str]:
    patterns = [
        r'Section\s+(\d+(?:\.\d+)*)',
        r'^(\d+(?:\.\d+)+)\s+[A-Z]',
        r'^\s*(\d+(?:\.\d+)+)\s',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def load_contract(json_path: str) -> List[Dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        pages = data
    elif isinstance(data, dict) and 'pages' in data:
        pages = data['pages']
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")
    return pages


def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase tokenization with basic legal-aware cleanup."""
    text = text.lower()
    text = re.sub(r'[^\w\s\.\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def create_embeddings(pages: List[Dict], model: SentenceTransformer, cache_path: Path) -> Tuple[List, np.ndarray, BM25Okapi]:
    chunks_with_meta = []
    for page in pages:
        page_num = page['page']
        text = page['text'].strip()
        if not text:
            continue
        section = extract_section_number(text)
        # Sentence-aware chunking preserves clause boundaries
        text_chunks = chunk_legal_document(text, strategy='hybrid', max_chunk_size=512)
        for chunk in text_chunks:
            chunks_with_meta.append({
                'page': page_num,
                'section': section,
                'text': chunk['text']
            })

    print(f"  Created {len(chunks_with_meta)} chunks from {len(pages)} pages")
    print(f"  Generating embeddings with {model._modules['0'].auto_model.config.name_or_path}...")
    texts = [c['text'] for c in chunks_with_meta]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build BM25 index for keyword scoring
    tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  Built BM25 keyword index")

    return chunks_with_meta, embeddings, bm25


def get_cache_path(json_path: str) -> Path:
    json_file = Path(json_path)
    return json_file.parent / f".{json_file.stem}_embeddings.pkl"


def load_or_create_embeddings(json_path: str, model_name: str) -> Tuple[List, np.ndarray, SentenceTransformer, BM25Okapi]:
    json_file = Path(json_path)
    cache_path = get_cache_path(json_path)
    use_cache = False

    if cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        json_mtime = json_file.stat().st_mtime
        if cache_mtime > json_mtime:
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                if cache_data.get('model_name') == model_name and 'bm25' in cache_data:
                    print(f"  Loading cached embeddings from {cache_path.name}")
                    use_cache = True
                    chunks = cache_data['chunks']
                    embeddings = cache_data['embeddings']
                    bm25 = cache_data['bm25']
            except Exception as e:
                print(f"  Cache load failed: {e}")

    if not use_cache:
        print(f"  Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        pages = load_contract(json_path)
        chunks, embeddings, bm25 = create_embeddings(pages, model, cache_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'model_name': model_name,
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'bm25': bm25
                }, f)
            print(f"  Saved embeddings + BM25 index to cache: {cache_path.name}")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")
    else:
        print(f"  Loading model: {model_name}")
        model = SentenceTransformer(model_name)

    return chunks, embeddings, model, bm25


def search(query: str, chunks: List[Dict], embeddings: np.ndarray,
           model: SentenceTransformer, top_n: int = 5,
           bm25: Optional[BM25Okapi] = None,
           semantic_weight: float = 0.7) -> List[Dict]:
    """
    Hybrid search combining semantic similarity and BM25 keyword matching.

    semantic_weight controls the balance:
      1.0 = pure semantic, 0.0 = pure keyword, 0.7 = default hybrid
    """
    # --- Semantic scoring ---
    model_name = model._modules['0'].auto_model.config.name_or_path.lower()
    if 'bge' in model_name:
        formatted_query = f"Represent this sentence for searching relevant passages: {query}"
    else:
        formatted_query = query

    query_embedding = model.encode([formatted_query], convert_to_numpy=True)[0]
    cos_similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    # Normalize semantic scores to [0, 1]
    sem_min, sem_max = cos_similarities.min(), cos_similarities.max()
    if sem_max > sem_min:
        semantic_scores = (cos_similarities - sem_min) / (sem_max - sem_min)
    else:
        semantic_scores = np.zeros_like(cos_similarities)

    # --- BM25 keyword scoring ---
    if bm25 is not None:
        query_tokens = tokenize_for_bm25(query)
        bm25_raw = bm25.get_scores(query_tokens)
        bm25_min, bm25_max = bm25_raw.min(), bm25_raw.max()
        if bm25_max > bm25_min:
            keyword_scores = (bm25_raw - bm25_min) / (bm25_max - bm25_min)
        else:
            keyword_scores = np.zeros_like(bm25_raw)

        # Combine: weighted sum
        combined_scores = (semantic_weight * semantic_scores
                          + (1 - semantic_weight) * keyword_scores)
    else:
        combined_scores = semantic_scores

    top_indices = np.argsort(combined_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        result = {
            'page': chunks[idx]['page'],
            'section': chunks[idx].get('section'),
            'score': float(combined_scores[idx]),
            'text': chunks[idx]['text']
        }
        if 'document' in chunks[idx]:
            result['document'] = chunks[idx]['document']
        results.append(result)

    return results


def interactive_search(json_path: str, model_name: str = "BAAI/bge-base-en-v1.5", top_n: int = 5):
    print(f"Hybrid Contract Search (Semantic + BM25)")
    print(f"  Loading: {Path(json_path).name}")
    chunks, embeddings, model, bm25 = load_or_create_embeddings(json_path, model_name)
    print(f"\nReady! Search across {len(chunks)} chunks from contract.")
    print(f"Type your query or 'quit' to exit.\n")

    while True:
        try:
            query = input("search> ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            results = search(query, chunks, embeddings, model, top_n, bm25=bm25)
            if not results:
                print("  No results found.\n")
                continue
            print(f"\nTop {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                score_pct = result['score'] * 100
                snippet = result['text'][:200].replace('\n', ' ')
                if len(result['text']) > 200:
                    snippet += "..."
                location = f"Page {result['page']}"
                if result.get('section'):
                    location += f", Section {result['section']}"
                print(f"{i}. {location} (relevance: {score_pct:.1f}%)")
                print(f"   {snippet}")
                print()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Semantic search over contract JSON")
    parser.add_argument("json_file", help="Path to extracted JSON file")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="Sentence transformer model name")
    parser.add_argument("--top", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    path = Path(args.json_file)
    if not path.exists():
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    interactive_search(str(path), model_name=args.model, top_n=args.top)


if __name__ == "__main__":
    main()
