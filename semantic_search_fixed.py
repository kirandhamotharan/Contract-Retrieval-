#!/usr/bin/env python3
"""
Semantic search over contract JSON using sentence-transformers embeddings.
Now with section number extraction and paragraph-based chunking!
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
except ImportError as e:
    print(f"Error: {e}")
    print("Install required packages: pip install sentence-transformers")
    sys.exit(1)


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


def chunk_text_by_paragraphs(text: str, max_chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into chunks by paragraphs, keeping related content together."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                sentences = re.split(r'[.!?]+\s+', current_chunk)
                if len(sentences) > 2:
                    current_chunk = '. '.join(sentences[-3:]) + '. ' + para
                else:
                    current_chunk = para
            else:
                words = para.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += (" " + word if temp_chunk else word)
                current_chunk = temp_chunk
        else:
            current_chunk += ("\n\n" + para if current_chunk else para)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def create_embeddings(pages: List[Dict], model: SentenceTransformer, cache_path: Path) -> Tuple[List, List]:
    chunks_with_meta = []
    for page in pages:
        page_num = page['page']
        text = page['text'].strip()
        if not text:
            continue
        section = extract_section_number(text)
        text_chunks = chunk_text_by_paragraphs(text)
        for chunk in text_chunks:
            chunks_with_meta.append({
                'page': page_num,
                'section': section,
                'text': chunk
            })
    
    print(f"  Created {len(chunks_with_meta)} chunks from {len(pages)} pages")
    print(f"  Generating embeddings with {model._modules['0'].auto_model.config.name_or_path}...")
    texts = [c['text'] for c in chunks_with_meta]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return chunks_with_meta, embeddings


def get_cache_path(json_path: str) -> Path:
    json_file = Path(json_path)
    return json_file.parent / f".{json_file.stem}_embeddings.pkl"


def load_or_create_embeddings(json_path: str, model_name: str) -> Tuple[List, np.ndarray, SentenceTransformer]:
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
                if cache_data.get('model_name') == model_name:
                    print(f"  Loading cached embeddings from {cache_path.name}")
                    use_cache = True
                    chunks = cache_data['chunks']
                    embeddings = cache_data['embeddings']
            except Exception as e:
                print(f"  Cache load failed: {e}")
    
    if not use_cache:
        print(f"  Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        pages = load_contract(json_path)
        chunks, embeddings = create_embeddings(pages, model, cache_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'model_name': model_name,
                    'chunks': chunks,
                    'embeddings': embeddings
                }, f)
            print(f"  Saved embeddings to cache: {cache_path.name}")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")
    else:
        print(f"  Loading model: {model_name}")
        model = SentenceTransformer(model_name)
    
    return chunks, embeddings, model


def search(query: str, chunks: List[Dict], embeddings: np.ndarray, model: SentenceTransformer, top_n: int = 5) -> List[Dict]:
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[::-1][:top_n * 3]
    page_results = {}
    
    for idx in top_indices:
        page_num = chunks[idx]['page']
        score = float(similarities[idx])
        if page_num not in page_results or score > page_results[page_num]['score']:
            page_results[page_num] = {
                'page': page_num,
                'section': chunks[idx].get('section'),
                'score': score,
                'text': chunks[idx]['text']
            }
    
    results = sorted(page_results.values(), key=lambda x: x['score'], reverse=True)[:top_n]
    return results


def interactive_search(json_path: str, model_name: str = "BAAI/bge-base-en-v1.5", top_n: int = 5):
    print(f"Semantic Contract Search")
    print(f"  Loading: {Path(json_path).name}")
    chunks, embeddings, model = load_or_create_embeddings(json_path, model_name)
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
            results = search(query, chunks, embeddings, model, top_n)
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
