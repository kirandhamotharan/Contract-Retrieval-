import streamlit as st
import json
from pathlib import Path
from semantic_search_fixed import load_or_create_embeddings, search, load_contract
import PyPDF2
import tempfile

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
    
    /* ===== EXPANDERS (RESULTS) ===== */
    .streamlit-expanderHeader {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500;
        background: var(--surface-gray);
        border-radius: 8px;
        padding: 1rem 1.25rem !important;
        border: 1px solid var(--border-gray);
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--light-blue);
        border-color: var(--primary-blue);
    }
    
    .streamlit-expanderContent {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1.25rem;
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
    
    /* ===== RESULT CARD ===== */
    .result-card {
        background: var(--surface-white);
        border: 1px solid var(--border-gray);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-blue);
        transition: all 0.2s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateX(2px);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .result-location {
        font-weight: 600;
        color: var(--primary-navy);
        font-size: 0.95rem;
    }
    
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
# MAIN LAYOUT
# =============================================================================
col_upload, col_spacer, col_search = st.columns([4, 0.5, 5.5])

# -----------------------------------------------------------------------------
# LEFT COLUMN: DOCUMENT UPLOAD
# -----------------------------------------------------------------------------
with col_upload:
    st.markdown("""
    <div class="content-card">
        <div class="section-title">
            📁 Document Source
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Contract Document",
        type=['pdf', 'json'],
        help="Supported formats: PDF (automatic text extraction) or pre-processed JSON",
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    
    use_default = st.checkbox(
        "Use demonstration contract (Ameritas Group Dental)",
        help="Load a sample contract to explore platform capabilities"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process uploaded file
    json_path = None
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            st.markdown('<div class="status-info">📄 Processing PDF document...</div>', unsafe_allow_html=True)
            contract_data = extract_pdf_to_json(uploaded_file)
            
            if contract_data:
                temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(contract_data, temp_json)
                temp_json.close()
                json_path = temp_json.name
                
                st.markdown(f"""
                <div class="content-card" style="margin-top: 1rem;">
                    <div class="status-ready">Document Loaded Successfully</div>
                    <div class="metrics-row">
                        <div class="metric-item">
                            <div class="metric-value">{len(contract_data)}</div>
                            <div class="metric-label">Pages</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">PDF</div>
                            <div class="metric-label">Format</div>
                        </div>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: var(--text-muted);">
                        <strong>Source:</strong> {uploaded_file.name}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        elif file_ext == 'json':
            temp_path = Path("temp_contract.json")
            temp_path.write_bytes(uploaded_file.read())
            json_path = str(temp_path)
            
            st.markdown(f"""
            <div class="content-card" style="margin-top: 1rem;">
                <div class="status-ready">Document Loaded Successfully</div>
                <p style="margin: 0.75rem 0 0 0; font-size: 0.85rem; color: var(--text-muted);">
                    <strong>Source:</strong> {uploaded_file.name}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    elif use_default:
        json_path = "DMS-2122-027CGroupDentalContract(Ameritas)_extracted.json"
        if Path(json_path).exists():
            st.markdown("""
            <div class="content-card" style="margin-top: 1rem;">
                <div class="status-ready">Demonstration Document Active</div>
                <p style="margin: 0.75rem 0 0 0; font-size: 0.85rem; color: var(--text-muted);">
                    <strong>Contract:</strong> Ameritas Group Dental Agreement
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Demonstration contract unavailable. Please upload a document.")
            json_path = None

# -----------------------------------------------------------------------------
# RIGHT COLUMN: SEARCH INTERFACE
# -----------------------------------------------------------------------------
with col_search:
    if json_path:
        st.markdown("""
        <div class="content-card">
            <div class="section-title">
                🔎 Semantic Search
            </div>
        """, unsafe_allow_html=True)
        
        @st.cache_resource
        def load_model(path):
            return load_or_create_embeddings(path, "sentence-transformers/all-mpnet-base-v2")
        
        @st.cache_data
        def get_contract_pages(path):
            return {page['page']: page['text'] for page in load_contract(path)}
        
        with st.spinner("Initializing AI analysis engine..."):
            chunks, embeddings, model = load_model(json_path)
            contract_pages = get_contract_pages(json_path)
        
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-item">
                <div class="metric-value">{len(chunks)}</div>
                <div class="metric-label">Indexed Sections</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{len(contract_pages)}</div>
                <div class="metric-label">Total Pages</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">Active</div>
                <div class="metric-label">AI Status</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form(key="search_form"):
            query = st.text_input(
                "Search Query",
                placeholder="Enter natural language query (e.g., 'What are the termination conditions?')",
                value=st.session_state.last_query,
                label_visibility="collapsed"
            )
            
            col_btn, col_slider = st.columns([1, 2])
            with col_btn:
                search_button = st.form_submit_button("Search", type="primary", use_container_width=True)
            with col_slider:
                top_n = st.slider("Results to display", 1, 10, 5, label_visibility="collapsed")
                st.caption(f"Returning top {top_n} results")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Execute search
        if search_button and query:
            with st.spinner("Analyzing document semantics..."):
                results = search(query, chunks, embeddings, model, top_n)
                st.session_state.search_results = results
                st.session_state.last_query = query
        
        # Display results
        if st.session_state.search_results:
            results = st.session_state.search_results
            
            if results:
                st.markdown(f"""
                <div class="content-card">
                    <div class="section-title">
                        📊 Analysis Results
                    </div>
                    <p style="margin: 0 0 1rem 0; color: var(--text-secondary);">
                        Found <strong>{len(results)}</strong> relevant sections matching your query
                    </p>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                for i, result in enumerate(results, 1):
                    score_pct = result['score'] * 100
                    page_num = result['page']
                    relevance_class = get_relevance_class(result['score'])
                    
                    location = f"Page {page_num}"
                    if result.get('section'):
                        location += f" · Section {result['section']}"
                    
                    with st.expander(
                        f"**Result {i}** — {location} — Relevance: {score_pct:.0f}%",
                        expanded=(i <= 2)
                    ):
                        st.markdown(f"**Extracted Content:**")
                        
                        display_text = result['text'][:600]
                        st.markdown(f"""
                        <div class="result-text">{display_text}{'...' if len(result['text']) > 600 else ''}</div>
                        """, unsafe_allow_html=True)
                        
                        if len(result['text']) > 600:
                            st.caption("Content truncated for display")
                        
                        st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
                        
                        show_full = st.checkbox(
                            f"View complete Page {page_num}",
                            key=f"show_{i}"
                        )
                        
                        if show_full:
                            st.markdown(f"**Complete Page {page_num} Content:**")
                            full_text = contract_pages.get(page_num, "Page content unavailable")
                            st.text_area(
                                "Full page content",
                                full_text,
                                height=350,
                                key=f"full_{i}",
                                label_visibility="collapsed"
                            )
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
                📄 Upload a document or select the demonstration contract to begin analysis
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