#!/usr/bin/env python3
"""
Semantic search over contract JSON using sentence-transformers embeddings.
Now with section number extraction!
"""

import sys
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse

# Check dependencies
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Install required packages: pip install sentence-transformers")
    sys.exit(1)


def extract_section_number(text: str) -> Optional[str]:
    """Extract section number from text (e.g., 'Section 4.2.1' or '4.2.1')."""
    # Match patterns like "Section 4.2.1", "Section 11.13", "4.2", etc.
    patterns = [
        r'Section\s+(\d+(?:\.\d+)*)',  # "Section 4.2.1"
        r'^(\d+(?:\.\d+)+)\s+[A-Z]',   # "4.2.1 Title"
        r'^\s*(\d+(?:\.\d+)+)\s',      # "  4.2 "
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def load_contract(json_path: str) -> List[Dict]:
    """Load contract pages from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both array and dict formats
    if isinstance(data, list):
        pages = data
    elif isinstance(data, dict) and 'pages' in data:
        pages = data['pages']
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")
    
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks


def create_embeddings(pages: List[Dict], model: SentenceTransformer, cache_path: Path) -> Tuple[List, List]:
    """Create embeddings for all page chunks with caching."""
    
    # Create chunks with metadata
    chunks_with_meta = []
    for page in pages:
        page_num = page['page']
        text = page['text'].strip()
        
        if not text:
            continue
        
        # Extract section number from page text
        section = extract_section_number(text)
        
        # Split into chunks
        text_chunks = chunk_text(text)
        
        for chunk in text_chunks:
            chunks_with_meta.append({
                'page': page_num,
                'section': section,
                'text': chunk
            })
    
    print(f"  Created {len(chunks_with_meta)} chunks from {len(pages)} pages")
    
    # Generate embeddings
    print(f"  Generating embeddings with {model._modules['0'].auto_model.config.name_or_path}...")
    texts = [c['text'] for c in chunks_with_meta]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return chunks_with_meta, embeddings


def get_cache_path(json_path: str) -> Path:
    """Generate cache file path based on JSON file."""
    json_file = Path(json_path)
    return json_file.parent / f".{json_file.stem}_embeddings.pkl"


def load_or_create_embeddings(json_path: str, model_name: str) -> Tuple[List, np.ndarray, SentenceTransformer]:
    """Load embeddings from cache or create new ones."""
    
    json_file = Path(json_path)
    cache_path = get_cache_path(json_path)
    
    # Check if cache is valid
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
        
        # Save to cache
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
    """Search for relevant chunks using cosine similarity."""
    
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Calculate cosine similarities
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_n * 3]  # Get extra for deduplication
    
    # Deduplicate by page (keep best score per page)
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
    
    # Sort by score and limit
    results = sorted(page_results.values(), key=lambda x: x['score'], reverse=True)[:top_n]
    
    return results


def interactive_search(json_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", top_n: int = 5):
    """Interactive search REPL."""
    
    print(f"Semantic Contract Search")
    print(f"  Loading: {Path(json_path).name}")
    
    # Load embeddings
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
            
            # Search
            results = search(query, chunks, embeddings, model, top_n)
            
            if not results:
                print("  No results found.\n")
                continue
            
            # Display results
            print(f"\nTop {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                score_pct = result['score'] * 100
                snippet = result['text'][:200].replace('\n', ' ')
                if len(result['text']) > 200:
                    snippet += "..."
                
                # Build location string
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
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    parser.add_argument("--top", type=int, default=5,
                       help="Number of results to return")
    
    args = parser.parse_args()
    
    path = Path(args.json_file)
    if not path.exists():
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    
    interactive_search(str(path), model_name=args.model, top_n=args.top)


if __name__ == "__main__":
    main()
