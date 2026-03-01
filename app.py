import streamlit as st
import json
from pathlib import Path
from semantic_search_fixed import load_or_create_embeddings, search, load_contract
import PyPDF2
import tempfile
import numpy as np

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

st.header("📁 Upload Contracts")

# Multiple file upload
uploaded_files = st.file_uploader(
    "Upload PDF or JSON contract files",
    type=['pdf', 'json'],
    accept_multiple_files=True,
    help="Upload one or more contracts (PDFs will be automatically extracted)"
)

use_default = st.checkbox("Use sample contract (Ameritas Dental)")

# Process uploaded files
processed_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            with st.spinner(f"📄 Extracting {uploaded_file.name}..."):
                contract_data = extract_pdf_to_json(uploaded_file)
                
                if contract_data:
                    temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                    json.dump(contract_data, temp_json)
                    temp_json.close()
                    processed_docs.append({
                        'name': uploaded_file.name,
                        'path': temp_json.name
                    })
                    
        elif file_ext == 'json':
            temp_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            temp_path.write(uploaded_file.read())
            temp_path.close()
            processed_docs.append({
                'name': uploaded_file.name,
                'path': temp_path.name
            })
    
    st.success(f"✅ Loaded {len(processed_docs)} document(s): {', '.join([d['name'] for d in processed_docs])}")
        
elif use_default:
    default_path = "DMS-2122-027CGroupDentalContract(Ameritas)_extracted.json"
    if Path(default_path).exists():
        processed_docs.append({
            'name': 'Ameritas Dental Contract',
            'path': default_path
        })
        st.success("✅ Using sample contract: Ameritas Dental")
    else:
        st.error("❌ Sample contract not found")

if processed_docs:
    st.header("🔎 Search")
    
    @st.cache_resource
    def load_multi_docs(doc_list):
        """Load and combine embeddings from multiple documents."""
        all_chunks = []
        all_embeddings = []
        model = None
        
        for doc in doc_list:
            chunks, embeddings, model = load_or_create_embeddings(doc['path'], "BAAI/bge-base-en-v1.5")
            
            # Add document name to each chunk
            for chunk in chunks:
                chunk['document'] = doc['name']
            
            all_chunks.extend(chunks)
            all_embeddings.append(embeddings)
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        return all_chunks, combined_embeddings, model
    
    @st.cache_data
    def get_all_contract_pages(doc_list):
        """Load all pages from all documents."""
        all_pages = {}
        for doc in doc_list:
            pages = {f"{doc['name']}:::{page['page']}": page['text'] 
                    for page in load_contract(doc['path'])}
            all_pages.update(pages)
        return all_pages
    
    with st.spinner("Loading AI model and indexing documents..."):
        chunks, embeddings, model = load_multi_docs(processed_docs)
        contract_pages = get_all_contract_pages(processed_docs)
    
    st.success(f"Ready! Indexed {len(chunks)} chunks across {len(processed_docs)} document(s)")
    
    with st.form(key="search_form"):
        query = st.text_area(
            "Enter your search queries (one per line):",
            placeholder="When does the contract end?\nHow are claims processed?\nWhat are the payment terms?",
            value=st.session_state.last_query,
            height=100
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.form_submit_button("🔍 Search", type="primary")
        with col2:
            top_n = st.slider("Results per query", 1, 10, 5)
    
    if search_button and query:
        queries = [q.strip() for q in query.split('\n') if q.strip()]
        
        with st.spinner(f"Searching {len(queries)} queries across {len(processed_docs)} document(s)..."):
            all_results = []
            for q in queries:
                results = search(q, chunks, embeddings, model, top_n)
                all_results.append({'query': q, 'results': results})
            
            st.session_state.search_results = all_results
            st.session_state.last_query = query
    
    if st.session_state.search_results:
        all_results = st.session_state.search_results
        
        for query_result in all_results:
            query_text = query_result['query']
            results = query_result['results']
            
            st.header(f"❓ {query_text}")
            
            if results:
                st.caption(f"{len(results)} matches found")
                
                for i, result in enumerate(results, 1):
                    score_pct = result['score'] * 100
                    page_num = result['page']
                    doc_name = result.get('document', 'Unknown')
                    
                    location = f"📄 {doc_name} - Page {page_num}"
                    if result.get('section'):
                        location += f", Section {result['section']}"
                    
                    with st.expander(
                        f"⭐ {i}. {location} — Relevance: {score_pct:.1f}%",
                        expanded=(i == 1)
                    ):
                        st.markdown(f"**Text snippet:**")
                        st.text(result['text'][:500])
                        if len(result['text']) > 500:
                            st.caption("... (truncated)")
                        
                        page_key = f"{doc_name}:::{page_num}"
                        show_full = st.checkbox(
                            f"�� Show Full Page {page_num}",
                            key=f"show_{query_text}_{i}"
                        )
                        
                        if show_full:
                            st.markdown("---")
                            st.markdown(f"**Complete text from {doc_name}, Page {page_num}:**")
                            full_text = contract_pages.get(page_key, "Page not found")
                            st.text_area(
                                "Full page",
                                full_text,
                                height=400,
                                key=f"full_{query_text}_{i}",
                                label_visibility="collapsed"
                            )
            else:
                st.warning("No results found for this query.")
            
            st.markdown("---")

else:
    st.info("👆 Upload PDF or JSON files, or use the sample to get started")

st.markdown("---")
st.caption("Built with Streamlit + Sentence Transformers | Phase C of Contract Retrieval Tool")
