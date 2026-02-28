import streamlit as st
import json
from pathlib import Path
from semantic_search_fixed import load_or_create_embeddings, search, load_contract
import PyPDF2
import tempfile

st.set_page_config(
    page_title="Contract Search Tool",
    page_icon="🔍",
    layout="wide"
)

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

def extract_pdf_to_json(pdf_file):
    pages = []
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        progress_bar = st.progress(0)
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            pages.append({"page": page_num + 1, "text": text})
            progress_bar.progress((page_num + 1) / total_pages)
        progress_bar.empty()
        return pages
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return None

st.title("🔍 Contract Search Tool")
st.markdown("AI-powered semantic search for legal documents")

st.header("📁 Upload Contract")
uploaded_file = st.file_uploader(
    "Upload a PDF or JSON contract file",
    type=['pdf', 'json'],
    help="Upload a PDF (will be automatically extracted) or pre-extracted JSON"
)

use_default = st.checkbox("Use sample contract (Ameritas Dental)")

json_path = None
if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if file_ext == 'pdf':
        st.info("📄 PDF detected - extracting text...")
        contract_data = extract_pdf_to_json(uploaded_file)
        
        if contract_data:
            temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(contract_data, temp_json)
            temp_json.close()
            json_path = temp_json.name
            st.success(f"✅ Extracted {len(contract_data)} pages from: {uploaded_file.name}")
        else:
            st.error("Failed to extract PDF")
            
    elif file_ext == 'json':
        temp_path = Path("temp_contract.json")
        temp_path.write_bytes(uploaded_file.read())
        json_path = str(temp_path)
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
elif use_default:
    json_path = "DMS-2122-027CGroupDentalContract(Ameritas)_extracted.json"
    if Path(json_path).exists():
        st.success("✅ Using sample contract: Ameritas Dental")
    else:
        st.error("❌ Sample contract not found")
        json_path = None

if json_path:
    st.header("🔎 Search")
    
    @st.cache_resource
    def load_model(path):
        return load_or_create_embeddings(path, "BAAI/bge-base-en-v1.5")
    
    @st.cache_data
    def get_contract_pages(path):
        return {page['page']: page['text'] for page in load_contract(path)}
    
    with st.spinner("Loading AI model..."):
        chunks, embeddings, model = load_model(json_path)
        contract_pages = get_contract_pages(json_path)
    
    st.success(f"Ready! Indexed {len(chunks)} chunks")
    
    with st.form(key="search_form"):
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., When does the contract end?",
            value=st.session_state.last_query
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.form_submit_button("🔍 Search", type="primary")
        with col2:
            top_n = st.slider("Number of results", 1, 10, 5)
    
    if search_button and query:
        with st.spinner("Searching..."):
            results = search(query, chunks, embeddings, model, top_n)
            st.session_state.search_results = results
            st.session_state.last_query = query
    
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        if results:
            st.header(f"📊 Results ({len(results)} matches found)")
            
            for i, result in enumerate(results, 1):
                score_pct = result['score'] * 100
                page_num = result['page']
                
                location = f"Page {page_num}"
                if result.get('section'):
                    location += f", Section {result['section']}"
                
                with st.expander(
                    f"⭐ {i}. {location} — Relevance: {score_pct:.1f}%",
                    expanded=(i <= 3)
                ):
                    st.markdown(f"**Text snippet:**")
                    st.text(result['text'][:500])
                    if len(result['text']) > 500:
                        st.caption("... (truncated)")
                    
                    show_full = st.checkbox(
                        f"📄 Show Full Page {page_num}",
                        key=f"show_{i}"
                    )
                    
                    if show_full:
                        st.markdown("---")
                        st.markdown(f"**Complete text from Page {page_num}:**")
                        full_text = contract_pages.get(page_num, "Page not found")
                        st.text_area(
                            "Full page",
                            full_text,
                            height=400,
                            key=f"full_{i}",
                            label_visibility="collapsed"
                        )
        else:
            st.warning("No results found. Try a different query.")

else:
    st.info("👆 Upload a PDF or JSON file, or use the sample to get started")

st.markdown("---")
st.caption("Built with Streamlit + Sentence Transformers | Phase C of Contract Retrieval Tool")
