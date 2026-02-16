"""
Semantic Chunking Module for Legal Documents
Chunks text by sentences/paragraphs to preserve context for embedding quality.

No external dependencies required - uses regex-based tokenization optimized
for legal text patterns.
"""

import re
from typing import List, Tuple


# Legal abbreviations that should NOT trigger sentence breaks
LEGAL_ABBREVIATIONS = {
    # Case citations
    'v', 'vs', 'no', 'nos', 'cf', 'see', 'e.g', 'i.e', 'et al', 'id', 'ibid',
    # Titles
    'mr', 'mrs', 'ms', 'dr', 'prof', 'hon', 'rev', 'gen', 'col', 'lt', 'sgt',
    # Corporate
    'inc', 'ltd', 'corp', 'llc', 'l.l.c', 'co', 'plc', 'llp', 'l.l.p',
    # Legal references
    'sec', 'sect', 'art', 'par', 'para', 'ch', 'pt', 'vol', 'app', 'ex',
    'u.s', 'u.s.c', 'c.f.r', 'f.r.d', 'f.2d', 'f.3d', 'f.supp', 's.ct',
    'u.s.c.a', 'fed', 'cir', 'dist', 'bankr', 'stat', 'reg', 'pub',
    # Common
    'st', 'ave', 'blvd', 'rd', 'sr', 'jr', 'esq', 'ph.d', 'j.d', 'm.d',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    # Legal Latin
    'viz', 'supra', 'infra', 'ante', 'post', 'op', 'cit', 'loc', 'passim',
}


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex-based tokenization.
    Optimized for legal documents with citations and abbreviations.
    """
    if not text or not text.strip():
        return []
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Protect abbreviations by replacing their periods temporarily
    protected = text
    
    # Protect numbered lists like "1." "2." "(a)." etc.
    protected = re.sub(r'(\d+)\.\s', r'\1███ ', protected)
    protected = re.sub(r'\(([a-zA-Z0-9]+)\)\.\s', r'(\1)███ ', protected)
    
    # Protect known abbreviations (case-insensitive)
    for abbrev in LEGAL_ABBREVIATIONS:
        # Match abbreviation followed by period and space/end
        pattern = re.compile(
            r'\b' + re.escape(abbrev).replace(r'\.', r'\.?') + r'\.\s',
            re.IGNORECASE
        )
        protected = pattern.sub(abbrev.upper() + '███ ', protected)
    
    # Protect initials like "J. Smith" or "A.B.C."
    protected = re.sub(r'\b([A-Z])\.\s*(?=[A-Z])', r'\1███', protected)
    
    # Protect decimal numbers like "3.5" or currency "$1.5M"
    protected = re.sub(r'(\d)\.(\d)', r'\1███\2', protected)
    
    # Protect section references like "Section 3.2.1"
    protected = re.sub(r'(\d)\.(\d)', r'\1███\2', protected)
    
    # Now split on sentence-ending punctuation followed by space and capital
    # or end of string
    sentence_pattern = re.compile(
        r'(?<=[.!?])'           # After sentence-ending punctuation
        r'(?:\s*["\'\)\]]*)'    # Optional closing quotes/parens
        r'\s+'                   # Required whitespace
        r'(?=[A-Z"\'\(\[]|$)'   # Before capital letter, quote, or end
    )
    
    raw_sentences = sentence_pattern.split(protected)
    
    # Restore protected periods
    sentences = []
    for sent in raw_sentences:
        restored = sent.replace('███', '.')
        restored = restored.strip()
        if restored:
            sentences.append(restored)
    
    # Merge very short fragments with previous sentence
    merged = []
    for sent in sentences:
        if merged and len(sent) < 30 and not sent[0].isupper():
            merged[-1] = merged[-1] + ' ' + sent
        else:
            merged.append(sent)
    
    return merged


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on double newlines or significant whitespace.
    """
    # Split on double newlines or multiple newlines
    paragraphs = re.split(r'\n\s*\n+', text)
    
    # Clean up and filter empty paragraphs
    cleaned = []
    for para in paragraphs:
        para = para.strip()
        para = re.sub(r'\s+', ' ', para)  # Normalize internal whitespace
        if para and len(para) > 20:  # Skip very short fragments
            cleaned.append(para)
    
    return cleaned


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 512,
    overlap_sentences: int = 2,
    min_chunk_size: int = 100
) -> List[dict]:
    """
    Chunk text by grouping sentences together up to max_chunk_size.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk (approximate)
        overlap_sentences: Number of sentences to overlap between chunks
        min_chunk_size: Minimum characters for a valid chunk
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_sentences = []
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_length = len(sent)
        
        # If adding this sentence exceeds max size, save current chunk
        if current_length + sent_length > max_chunk_size and current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text) >= min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'sentence_count': len(current_sentences),
                    'char_count': len(chunk_text)
                })
            
            # Keep overlap sentences for context continuity
            if overlap_sentences > 0 and len(current_sentences) > overlap_sentences:
                current_sentences = current_sentences[-overlap_sentences:]
                current_length = sum(len(s) for s in current_sentences)
            else:
                current_sentences = []
                current_length = 0
        
        current_sentences.append(sent)
        current_length += sent_length + 1  # +1 for space
        i += 1
    
    # Don't forget the last chunk
    if current_sentences:
        chunk_text = ' '.join(current_sentences)
        if len(chunk_text) >= min_chunk_size:
            chunks.append({
                'text': chunk_text,
                'sentence_count': len(current_sentences),
                'char_count': len(chunk_text)
            })
    
    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chunk_size: int = 1024,
    overlap_paragraphs: int = 1,
    min_chunk_size: int = 100
) -> List[dict]:
    """
    Chunk text by grouping paragraphs together up to max_chunk_size.
    Better for documents with clear paragraph structure.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap_paragraphs: Number of paragraphs to overlap
        min_chunk_size: Minimum characters for a valid chunk
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    paragraphs = split_into_paragraphs(text)
    
    if not paragraphs:
        # Fall back to sentence chunking if no clear paragraphs
        return chunk_by_sentences(text, max_chunk_size)
    
    chunks = []
    current_paragraphs = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # If single paragraph exceeds max, chunk it by sentences
        if para_length > max_chunk_size:
            # Save current chunk first
            if current_paragraphs:
                chunk_text = '\n\n'.join(current_paragraphs)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'paragraph_count': len(current_paragraphs),
                        'char_count': len(chunk_text)
                    })
                current_paragraphs = []
                current_length = 0
            
            # Chunk the long paragraph by sentences
            sub_chunks = chunk_by_sentences(para, max_chunk_size)
            chunks.extend(sub_chunks)
            continue
        
        # If adding this paragraph exceeds max, save current chunk
        if current_length + para_length > max_chunk_size and current_paragraphs:
            chunk_text = '\n\n'.join(current_paragraphs)
            if len(chunk_text) >= min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'paragraph_count': len(current_paragraphs),
                    'char_count': len(chunk_text)
                })
            
            # Keep overlap for continuity
            if overlap_paragraphs > 0 and len(current_paragraphs) > overlap_paragraphs:
                current_paragraphs = current_paragraphs[-overlap_paragraphs:]
                current_length = sum(len(p) for p in current_paragraphs)
            else:
                current_paragraphs = []
                current_length = 0
        
        current_paragraphs.append(para)
        current_length += para_length + 2  # +2 for \n\n
    
    # Last chunk
    if current_paragraphs:
        chunk_text = '\n\n'.join(current_paragraphs)
        if len(chunk_text) >= min_chunk_size:
            chunks.append({
                'text': chunk_text,
                'paragraph_count': len(current_paragraphs),
                'char_count': len(chunk_text)
            })
    
    return chunks


def chunk_legal_document(
    text: str,
    strategy: str = 'hybrid',
    max_chunk_size: int = 512,
    overlap: int = 2
) -> List[dict]:
    """
    Main chunking function for legal documents.
    
    Args:
        text: Document text to chunk
        strategy: 'sentences', 'paragraphs', or 'hybrid' (default)
        max_chunk_size: Maximum characters per chunk
        overlap: Number of units (sentences or paragraphs) to overlap
    
    Returns:
        List of chunk dictionaries
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    if strategy == 'sentences':
        return chunk_by_sentences(text, max_chunk_size, overlap)
    elif strategy == 'paragraphs':
        return chunk_by_paragraphs(text, max_chunk_size, overlap)
    elif strategy == 'hybrid':
        # Use paragraphs if document has clear structure, else sentences
        paragraphs = split_into_paragraphs(text)
        avg_para_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        
        # If paragraphs are reasonably sized, use paragraph chunking
        if len(paragraphs) > 3 and 100 < avg_para_length < max_chunk_size:
            return chunk_by_paragraphs(text, max_chunk_size, overlap)
        else:
            return chunk_by_sentences(text, max_chunk_size, overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'sentences', 'paragraphs', or 'hybrid'")


def chunk_contract_pages(
    pages: List[dict],
    strategy: str = 'hybrid',
    max_chunk_size: int = 512,
    overlap: int = 2
) -> List[dict]:
    """
    Chunk a list of contract pages (from PDF extraction).
    Preserves page number metadata.
    
    Args:
        pages: List of dicts with 'page' and 'text' keys
        strategy: Chunking strategy
        max_chunk_size: Maximum chunk size
        overlap: Overlap units
    
    Returns:
        List of chunks with page metadata
    """
    all_chunks = []
    
    for page_data in pages:
        page_num = page_data.get('page', 0)
        page_text = page_data.get('text', '')
        
        if not page_text.strip():
            continue
        
        page_chunks = chunk_legal_document(
            page_text,
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            overlap=overlap
        )
        
        for chunk in page_chunks:
            chunk['page'] = page_num
            all_chunks.append(chunk)
    
    return all_chunks


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Backward-compatible wrapper that returns list of strings.
    Internally uses semantic chunking.
    
    Args:
        text: Text to chunk
        chunk_size: Max characters per chunk (maps to max_chunk_size)
        overlap: Ignored in new implementation; uses 2-sentence overlap
    
    Returns:
        List of chunk text strings
    """
    chunks = chunk_legal_document(
        text,
        strategy='hybrid',
        max_chunk_size=chunk_size,
        overlap=2
    )
    return [c['text'] for c in chunks]


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == '__main__':
    sample_legal_text = """
    WHEREAS, the Parties wish to enter into an agreement governing the terms 
    and conditions of their business relationship; and
    
    WHEREAS, MetLife Insurance Company (hereinafter "Company") provides 
    insurance products and services pursuant to applicable regulations under 
    U.S.C. § 1234 and related provisions; and
    
    NOW, THEREFORE, in consideration of the mutual covenants contained herein, 
    the Parties agree as follows:
    
    1. DEFINITIONS. For purposes of this Agreement, the following terms shall 
    have the meanings set forth below. "Affiliate" means any entity that 
    directly or indirectly controls, is controlled by, or is under common 
    control with a Party. "Confidential Information" means all non-public 
    information disclosed by either Party to the other Party, whether orally, 
    in writing, or by inspection of tangible objects.
    
    2. TERM AND TERMINATION. This Agreement shall commence on the Effective 
    Date and shall continue for a period of three (3) years unless earlier 
    terminated in accordance with Section 2.2. Either Party may terminate 
    this Agreement upon sixty (60) days prior written notice to the other 
    Party. Upon termination, all rights and obligations shall cease, except 
    those provisions which by their nature should survive termination.
    
    3. GOVERNING LAW. This Agreement shall be governed by and construed in 
    accordance with the laws of the State of New York, without regard to its 
    conflicts of law principles. Any disputes arising under this Agreement 
    shall be resolved through binding arbitration in accordance with the 
    rules of the American Arbitration Association.
    """
    
    print("=" * 60)
    print("SEMANTIC CHUNKING DEMO")
    print("=" * 60)
    
    # Test sentence chunking
    print("\n--- Sentence-based chunks ---")
    sentence_chunks = chunk_legal_document(sample_legal_text, strategy='sentences', max_chunk_size=400)
    for i, chunk in enumerate(sentence_chunks, 1):
        print(f"\nChunk {i} ({chunk['char_count']} chars, {chunk['sentence_count']} sentences):")
        print(f"  {chunk['text'][:100]}...")
    
    # Test paragraph chunking
    print("\n\n--- Paragraph-based chunks ---")
    para_chunks = chunk_legal_document(sample_legal_text, strategy='paragraphs', max_chunk_size=600)
    for i, chunk in enumerate(para_chunks, 1):
        print(f"\nChunk {i} ({chunk['char_count']} chars):")
        print(f"  {chunk['text'][:100]}...")
    
    # Test hybrid
    print("\n\n--- Hybrid chunks (auto-detect) ---")
    hybrid_chunks = chunk_legal_document(sample_legal_text, strategy='hybrid', max_chunk_size=500)
    for i, chunk in enumerate(hybrid_chunks, 1):
        print(f"\nChunk {i} ({chunk['char_count']} chars):")
        print(f"  {chunk['text'][:100]}...")
    
    print("\n\nTotal chunks:", len(hybrid_chunks))