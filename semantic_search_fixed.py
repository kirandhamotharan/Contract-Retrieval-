#!/usr/bin/env python3
"""
Semantic search over contract JSON using sentence-transformers embeddings.
- Legal domain model support (Free Law Project)
- Semantic chunking (sentence/paragraph-aware)
- Section number extraction
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

# Import semantic chunking
try:
    from semantic_chunking import chunk_contract_pages, chunk_legal_document
except ImportError:
    print("Warning: semantic_chunking not found, using fallback chunking")
    chunk_contract_pages = None
    chunk_legal_document = None


# =============================================================================
# CONFIGURATION
# =============================================================================

# Available models with their configurations
MODEL_CONFIGS = {
    # General purpose (fast, good baseline)
    "sentence-transformers/all-MiniLM-L6-v2": {
        "query_prefix": "",
        "document_prefix": "",
        "max_seq_length": 256,
    },
    # Higher quality general purpose
    "sentence-transformers/all-mpnet-base-v2": {
        "query_prefix": "",
        "document_prefix": "",
        "max_seq_length": 384,
    },
    # Legal domain - Free Law Project (RECOMMENDED)
    "Free-Law-Project/modernbert-embed-base_finetune_512": {
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "max_seq_length": 512,
    },
    # Legal domain - longer context
    "Free-Law-Project/modernbert-embed-base_finetune_8192": {
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "max_seq_length": 8192,
    },
    # Legal domain - Matryoshka (flexible dimensions)
    "AdamLucek/ModernBERT-embed-base-legal-MRL": {
        "query_prefix": "",
        "document_prefix": "",
        "max_seq_length": 8192,
    },
}

# Default model
DEFAULT_MODEL = "Free-Law-Project/modernbert-embed-base_finetune_512"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_config(model_name: str) -> Dict:
    """Get configuration for a model, with sensible defaults for unknown models."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Default config for unknown models
    return {
        "query_prefix": "",
        "document_prefix": "",
        "max_seq_length": 512,
    }


def extract_section_number(text: str) -> Optional[str]:
    """Extract section number from text (e.g., 'Section 4.2.1' or '4.2.1')."""
    patterns = [
        r'Section\s+(\d+(?:\.\d+)*)',  # "Section 4.2.1"
        r'SECTION\s+(\d+(?:\.\d+)*)',  # "SECTION 4.2.1"
        r'Article\s+(\d+(?:\.\d+)*)',  # "Article 4"
        r'ARTICLE\s+(\d+(?:\.\d+)*)',  # "ARTICLE 4"
        r'^(\d+(?:\.\d+)+)\s+[A-Z]',   # "4.2.1 Title"
        r'^\s*(\d+(?:\.\d+)+)\s',      # "  4.2 "
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
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


# =============================================================================
# FALLBACK CHUNKING (if semantic_chunking not available)
# =============================================================================

def fallback_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Simple character-based chunking as fallback."""
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


# =============================================================================
# EMBEDDING CREATION
# =============================================================================

def create_embeddings(
    pages: List[Dict],
    model: SentenceTransformer,
    model_name: str,
    cache_path: Path
) -> Tuple[List[Dict], np.ndarray]:
    """Create embeddings for all page chunks with semantic chunking."""
    
    config = get_model_config(model_name)
    doc_prefix = config["document_prefix"]
    max_chunk_size = min(config["max_seq_length"] * 4, 2048)  # Approximate chars
    
    # Use semantic chunking if available
    if chunk_contract_pages is not None:
        print(f"  Using semantic chunking (sentence/paragraph-aware)...")
        chunks_with_meta = chunk_contract_pages(
            pages,
            strategy='hybrid',
            max_chunk_size=max_chunk_size,
            overlap=2
        )
        
        # Add section numbers to chunks
        for chunk in chunks_with_meta:
            if 'section' not in chunk or chunk.get('section') is None:
                chunk['section'] = extract_section_number(chunk['text'])
    else:
        # Fallback to simple chunking
        print(f"  Using fallback chunking (character-based)...")
        chunks_with_meta = []
        for page in pages:
            page_num = page['page']
            text = page['text'].strip()
            
            if not text:
                continue
            
            section = extract_section_number(text)
            text_chunks = fallback_chunk_text(text, chunk_size=max_chunk_size)
            
            for chunk in text_chunks:
                chunks_with_meta.append({
                    'page': page_num,
                    'section': section,
                    'text': chunk
                })
    
    print(f"  Created {len(chunks_with_meta)} chunks from {len(pages)} pages")
    
    # Prepare texts with document prefix for legal models
    if doc_prefix:
        print(f"  Applying document prefix for legal model...")
        texts = [doc_prefix + c['text'] for c in chunks_with_meta]
    else:
        texts = [c['text'] for c in chunks_with_meta]
    
    # Generate embeddings
    print(f"  Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Pre-normalize for faster search
    )
    
    return chunks_with_meta, embeddings


def get_cache_path(json_path: str, model_name: str) -> Path:
    """Generate cache file path based on JSON file and model."""
    json_file = Path(json_path)
    # Include model name hash in cache to avoid mixing embeddings
    model_hash = abs(hash(model_name)) % 10000
    return json_file.parent / f".{json_file.stem}_emb_{model_hash}.pkl"


def load_or_create_embeddings(
    json_path: str,
    model_name: str = DEFAULT_MODEL
) -> Tuple[List[Dict], np.ndarray, SentenceTransformer]:
    """Load embeddings from cache or create new ones."""
    
    json_file = Path(json_path)
    cache_path = get_cache_path(json_path, model_name)
    
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
        chunks, embeddings = create_embeddings(pages, model, model_name, cache_path)
        
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
    
    # Store model name for later reference
    model._model_name = model_name
    
    return chunks, embeddings, model


# =============================================================================
# SEARCH
# =============================================================================

def search(
    query: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_n: int = 5,
    deduplicate_pages: bool = True
) -> List[Dict]:
    """Search for relevant chunks using cosine similarity."""
    
    # Get model config for query prefix
    model_name = getattr(model, '_model_name', '')
    config = get_model_config(model_name)
    query_prefix = config["query_prefix"]
    
    # Encode query with prefix
    query_text = query_prefix + query if query_prefix else query
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    
    # Calculate cosine similarities (embeddings already normalized)
    similarities = np.dot(embeddings, query_embedding)
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_n * 3]
    
    if deduplicate_pages:
        # Keep best score per page
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
    else:
        # Return all top chunks
        results = []
        for idx in top_indices[:top_n]:
            results.append({
                'page': chunks[idx]['page'],
                'section': chunks[idx].get('section'),
                'score': float(similarities[idx]),
                'text': chunks[idx]['text']
            })
    
    return results


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def interactive_search(
    json_path: str,
    model_name: str = DEFAULT_MODEL,
    top_n: int = 5
):
    """Interactive search REPL."""
    
    print(f"\n{'='*60}")
    print(f"ContractIQ Semantic Search")
    print(f"{'='*60}")
    print(f"  Document: {Path(json_path).name}")
    print(f"  Model: {model_name}")
    
    # Load embeddings
    chunks, embeddings, model = load_or_create_embeddings(json_path, model_name)
    
    print(f"\n✓ Ready! Indexed {len(chunks)} semantic chunks.")
    print(f"  Type your query or 'quit' to exit.\n")
    
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
            print(f"\n{'─'*50}")
            print(f"Top {len(results)} results for: \"{query}\"")
            print(f"{'─'*50}\n")
            
            for i, result in enumerate(results, 1):
                score_pct = result['score'] * 100
                
                # Clean snippet
                snippet = result['text'][:250].replace('\n', ' ')
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                if len(result['text']) > 250:
                    snippet += "..."
                
                # Build location string
                location = f"Page {result['page']}"
                if result.get('section'):
                    location += f" · Section {result['section']}"
                
                # Relevance indicator
                if score_pct >= 70:
                    relevance = "●●●"
                elif score_pct >= 50:
                    relevance = "●●○"
                else:
                    relevance = "●○○"
                
                print(f"{i}. [{relevance}] {location} ({score_pct:.0f}%)")
                print(f"   {snippet}")
                print()
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over contract documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  --model sentence-transformers/all-MiniLM-L6-v2       (fast, general)
  --model sentence-transformers/all-mpnet-base-v2     (better quality)
  --model Free-Law-Project/modernbert-embed-base_finetune_512  (legal domain, RECOMMENDED)
  --model AdamLucek/ModernBERT-embed-base-legal-MRL   (legal, flexible dims)
        """
    )
    parser.add_argument("json_file", help="Path to extracted JSON file")
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--top", 
        type=int, 
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    args = parser.parse_args()
    
    path = Path(args.json_file)
    if not path.exists():
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    
    interactive_search(str(path), model_name=args.model, top_n=args.top)


if __name__ == "__main__":
    main()