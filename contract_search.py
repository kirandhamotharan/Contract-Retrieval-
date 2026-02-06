import PyPDF2
import re

def extract_contract_text(pdf_path):
    """Extract text from PDF with page numbers"""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pages_text = []
        
        for page_num in range(len(pdf_reader.pages)):
            text = pdf_reader.pages[page_num].extract_text()
            pages_text.append({
                'page': page_num + 1,
                'text': text
            })
        
        return pages_text

def search_contract(pages_text, query):
    """Search for keywords in contract"""
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
    """Get text snippet around first keyword match"""
    text_lower = text.lower()
    
    for keyword in keywords:
        pos = text_lower.find(keyword)
        if pos != -1:
            start = max(0, pos - context_chars)
            end = min(len(text), pos + context_chars)
            snippet = text[start:end]
            return f"...{snippet}..."
    
    return text[:400] + "..."

# Ask user for query
print("Contract Search Tool")
print("=" * 50)
query = input("Enter your search query: ")

pdf_path = 'DMS-2122-027CGroupDentalContract(Ameritas).pdf'

print(f"\nSearching for: '{query}'")
print("Loading contract...")
pages_text = extract_contract_text(pdf_path)
print(f"Loaded {len(pages_text)} pages\n")

results = search_contract(pages_text, query)

if results:
    print(f"Found {len(results)} pages with matches\n")
    print("=" * 70)
    
    # Show top 5 most relevant pages
    for i, result in enumerate(results[:5], 1):
        print(f"\n[Result {i}] Page {result['page']} - {result['matches']} keyword matches")
        print(f"Preview: {result['preview']}")
        print("-" * 70)
else:
    print("No results found")
