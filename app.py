import streamlit as st
import json
import html
from pathlib import Path
from semantic_search_fixed import load_or_create_embeddings, search, load_contract
import PyPDF2
import tempfile
import numpy as np

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ContractIQ | Enterprise Document Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# EXECUTIVE-GRADE CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    /* ===== FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary-navy: #0a2540;
        --primary-blue: #0066cc;
        --accent-blue: #0077c8;
        --light-blue: #e8f4fc;
        --surface-white: #ffffff;
        --surface-gray: #f8f9fa;
        --border-gray: #e5e7eb;
        --text-primary: #1a1a2e;
        --text-secondary: #4a5568;
        --text-muted: #718096;
        --success-green: #059669;
        --warning-amber: #d97706;
    }

    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }

    .main .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1400px;
    }

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: var(--primary-navy) !important;
        letter-spacing: -0.02em;
    }

    p, span, div, label {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ===== HEADER SECTION ===== */
    .executive-header {
        background: linear-gradient(135deg, var(--primary-navy) 0%, #1e3a5f 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 24px rgba(10, 37, 64, 0.12);
        position: relative;
        overflow: hidden;
    }

    .executive-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 400px;
        height: 100%;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.03) 100%);
        pointer-events: none;
    }

    .executive-header h1 {
        color: #ffffff !important;
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.03em;
    }

    .executive-header .tagline {
        color: rgba(255,255,255,0.75);
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0;
        letter-spacing: 0.01em;
    }

    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #ffffff;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        backdrop-filter: blur(4px);
    }

    /* ===== CARDS & CONTAINERS ===== */
    .content-card {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }

    .content-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .section-title {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: var(--primary-navy) !important;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--light-blue);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ===== STATUS INDICATORS ===== */
    .status-ready {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: var(--success-green);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid rgba(5, 150, 105, 0.2);
    }

    .status-ready::before {
        content: '●';
        font-size: 0.6rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .status-info {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--light-blue);
        color: var(--primary-blue);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid rgba(0, 102, 204, 0.15);
    }

    /* ===== METRICS ROW ===== */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin: 1.25rem 0;
    }

    .metric-item {
        flex: 1;
        background: var(--surface-gray);
        padding: 1rem 1.25rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--border-gray);
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--primary-navy);
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* ===== FORM ELEMENTS ===== */
    .stTextInput > div > div > input {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 1rem;
        padding: 0.875rem 1rem;
        border: 2px solid var(--border-gray);
        border-radius: 8px;
        background: var(--surface-white);
        transition: all 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted);
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease;
        letter-spacing: 0.01em;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--accent-blue) 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.25);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.35);
    }

    /* ===== RESULT TOGGLE BUTTON ===== */
    .stButton > button[kind="secondary"] {
        background: var(--surface-gray);
        color: var(--primary-navy);
        border: 1px solid var(--border-gray);
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
    }

    .stButton > button[kind="secondary"]:hover {
        background: var(--light-blue);
        border-color: var(--primary-blue);
        color: var(--primary-blue);
    }

    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        background: var(--surface-gray);
        border: 2px dashed var(--border-gray);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.2s ease;
    }

    .stFileUploader:hover {
        border-color: var(--primary-blue);
        background: var(--light-blue);
    }

    .stFileUploader label {
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }

    /* ===== RESULT HEADER ROW ===== */
    .result-header-bar {
        background: var(--surface-gray);
        border: 1px solid var(--border-gray);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-weight: 500;
        color: var(--primary-navy);
        font-size: 0.95rem;
        display: flex;
        align-items: center;
    }

    .result-header-bar:hover {
        background: var(--light-blue);
        border-color: var(--primary-blue);
    }

    /* ===== RESULT CONTENT PANEL ===== */
    .result-content-panel {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1.25rem;
        margin-bottom: 0.5rem;
    }

    /* ===== RELEVANCE SCORE BADGE ===== */
    .relevance-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .relevance-high {
        background: #dcfce7;
        color: #166534;
    }

    .relevance-medium {
        background: #fef3c7;
        color: #92400e;
    }

    .relevance-low {
        background: #f3f4f6;
        color: #6b7280;
    }

    /* ===== RESULT TEXT ===== */
    .result-text {
        background: var(--surface-gray);
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        line-height: 1.6;
        color: var(--text-secondary);
        border: 1px solid var(--border-gray);
    }

    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--accent-blue) 100%);
        border-radius: 4px;
    }

    /* ===== SLIDER ===== */
    .stSlider > div > div > div > div {
        background: var(--primary-blue);
    }

    /* ===== CHECKBOX ===== */
    .stCheckbox label span {
        font-weight: 400;
        color: var(--text-secondary);
    }

    /* ===== TEXTAREA ===== */
    .stTextArea textarea {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.875rem;
        background: var(--surface-gray);
        border: 1px solid var(--border-gray);
        border-radius: 6px;
    }

    /* ===== QUERY SECTION HEADER ===== */
    .query-header {
        background: linear-gradient(135deg, var(--light-blue) 0%, #dbeafe 100%);
        border: 1px solid rgba(0, 102, 204, 0.15);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--primary-navy);
    }

    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 8px;
        border: none;
    }

    /* ===== FOOTER ===== */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-gray);
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def extract_pdf_to_json(pdf_file):
    """Extract text from PDF and return JSON format."""
    pages = []
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)

        progress_bar = st.progress(0, text="Processing document...")
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            pages.append({"page": page_num + 1, "text": text})
            progress_bar.progress(
                (page_num + 1) / total_pages,
                text=f"Analyzing page {page_num + 1} of {total_pages}..."
            )

        progress_bar.empty()
        return pages
    except Exception as e:
        st.error(f"Document processing error: {e}")
        return None

def get_relevance_class(score):
    """Return CSS class based on relevance score."""
    if score >= 0.7:
        return "relevance-high"
    elif score >= 0.4:
        return "relevance-medium"
    return "relevance-low"

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="executive-header">
    <div class="header-badge">Enterprise Intelligence Platform</div>
    <h1>ContractIQ</h1>
    <p class="tagline">AI-Powered Semantic Analysis for Legal Documents</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# DOCUMENT UPLOAD SECTION
# =============================================================================
st.markdown("""
<div class="content-card">
    <div class="section-title">
        Document Source
    </div>
</div>
""", unsafe_allow_html=True)

col_upload, col_status = st.columns([3, 2])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload Contract Documents",
        type=['pdf', 'json'],
        accept_multiple_files=True,
        help="Supported formats: PDF (automatic text extraction) or pre-processed JSON. Upload one or more contracts.",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    use_default = st.checkbox(
        "Use demonstration contract (Ameritas Group Dental)",
        help="Load a sample contract to explore platform capabilities"
    )

# Process uploaded files
processed_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext == 'pdf':
            with st.spinner(f"Processing {uploaded_file.name}..."):
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

    with col_status:
        doc_names = ", ".join([html.escape(d['name']) for d in processed_docs])
        st.markdown(f"""
        <div class="content-card" style="margin-top: 0;">
            <div class="status-ready">Documents Loaded Successfully</div>
            <div class="metrics-row">
                <div class="metric-item">
                    <div class="metric-value">{len(processed_docs)}</div>
                    <div class="metric-label">Documents</div>
                </div>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: var(--text-muted);">
                <strong>Sources:</strong> {doc_names}
            </p>
        </div>
        """, unsafe_allow_html=True)

elif use_default:
    default_path = "DMS-2122-027CGroupDentalContract(Ameritas)_extracted.json"
    if Path(default_path).exists():
        processed_docs.append({
            'name': 'Ameritas Dental Contract',
            'path': default_path
        })
        with col_status:
            st.markdown("""
            <div class="content-card" style="margin-top: 0;">
                <div class="status-ready">Demonstration Document Active</div>
                <p style="margin: 0.75rem 0 0 0; font-size: 0.85rem; color: var(--text-muted);">
                    <strong>Contract:</strong> Ameritas Group Dental Agreement
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Demonstration contract unavailable. Please upload a document.")

# =============================================================================
# SEARCH SECTION
# =============================================================================
if processed_docs:
    st.markdown("""
    <div class="content-card">
        <div class="section-title">
            Semantic Search
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    with st.spinner("Initializing AI analysis engine..."):
        chunks, embeddings, model = load_multi_docs(processed_docs)
        contract_pages = get_all_contract_pages(processed_docs)

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-item">
            <div class="metric-value">{len(chunks)}</div>
            <div class="metric-label">Indexed Sections</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{len(processed_docs)}</div>
            <div class="metric-label">Documents</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">Active</div>
            <div class="metric-label">AI Status</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form(key="search_form"):
        query = st.text_area(
            "Enter your search queries (one per line):",
            placeholder="What are the termination conditions?\nHow are claims processed?\nWhat are the payment terms?",
            value=st.session_state.last_query,
            height=100
        )

        col_btn, col_slider = st.columns([1, 2])
        with col_btn:
            search_button = st.form_submit_button("Search", type="primary", use_container_width=True)
        with col_slider:
            top_n = st.slider("Results per query", 1, 10, 5, label_visibility="collapsed")
            st.caption(f"Returning top {top_n} results per query")

    # Execute search
    if search_button and query:
        queries = [q.strip() for q in query.split('\n') if q.strip()]

        with st.spinner(f"Analyzing {len(queries)} queries across {len(processed_docs)} document(s)..."):
            all_results = []
            for q in queries:
                results = search(q, chunks, embeddings, model, top_n)
                all_results.append({'query': q, 'results': results})

            st.session_state.search_results = all_results
            st.session_state.last_query = query

    # Display results
    if st.session_state.search_results:
        all_results = st.session_state.search_results

        for query_result in all_results:
            query_text = query_result['query']
            results = query_result['results']

            st.markdown(f"""
            <div class="query-header">
                {html.escape(query_text)}
            </div>
            """, unsafe_allow_html=True)

            if results:
                st.markdown(f"""
                <p style="margin: 0 0 1rem 0; color: var(--text-secondary);">
                    Found <strong>{len(results)}</strong> relevant sections
                </p>
                """, unsafe_allow_html=True)

                for i, result in enumerate(results, 1):
                    score_pct = result['score'] * 100
                    page_num = result['page']
                    doc_name = result.get('document', 'Unknown')
                    relevance_class = get_relevance_class(result['score'])

                    location = f"{html.escape(doc_name)} — Page {page_num}"
                    if result.get('section'):
                        location += f" · Section {result['section']}"

                    # Result header bar
                    st.markdown(f"""
                    <div class="result-header-bar">
                        <strong>Result {i}</strong>&nbsp;&nbsp;—&nbsp;&nbsp;{location}&nbsp;&nbsp;—&nbsp;&nbsp;
                        <span class="relevance-badge {relevance_class}">{score_pct:.0f}% Relevance</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Result content
                    st.markdown("**Extracted Content:**")

                    display_text = html.escape(result['text'][:600])
                    st.markdown(f"""
                    <div class="result-text">{display_text}{'...' if len(result['text']) > 600 else ''}</div>
                    """, unsafe_allow_html=True)

                    if len(result['text']) > 600:
                        st.caption("Content truncated for display")

                    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

                    page_key = f"{doc_name}:::{page_num}"
                    show_full = st.checkbox(
                        f"View complete Page {page_num} from {doc_name}",
                        key=f"show_{query_text}_{i}"
                    )

                    if show_full:
                        st.markdown(f"**Complete Page {page_num} Content:**")
                        full_text = contract_pages.get(page_key, "Page content unavailable")
                        st.text_area(
                            "Full page content",
                            full_text,
                            height=350,
                            key=f"full_{query_text}_{i}",
                            label_visibility="collapsed"
                        )

                    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="content-card">
                    <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
                        <p style="font-size: 1.1rem; margin: 0;">No matching sections found</p>
                        <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                            Try refining your query or using different terminology
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="content-card" style="text-align: center; padding: 3rem;">
        <p style="font-size: 1.25rem; color: var(--text-secondary); margin: 0;">
            Upload a document or select the demonstration contract to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <p>ContractIQ Enterprise Document Intelligence · Powered by AI Semantic Analysis</p>
</div>
""", unsafe_allow_html=True)
