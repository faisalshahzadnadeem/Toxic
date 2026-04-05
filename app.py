# app.py
import streamlit as st
import time
from datetime import datetime
from rag_engine import ingest_documents, hybrid_retrieve, generate_explainable_response
from safety_layer import safety_triage

# Page configuration
st.set_page_config(
    page_title="Toxic Guard - AI Poisoning Triage",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Cache the document loading so it only runs once per session
@st.cache_resource(show_spinner="Loading toxicology data into vector database...")
def load_vector_store():
    try:
        index, chunks = ingest_documents()
        return index, chunks
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None, None

index, chunks = load_vector_store()

# Header
st.title("🛡️ Toxic Guard")
st.markdown("### AI-Powered Poisoning Triage Recommendation System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    if index is not None:
        st.success(f"✅ Vector Database Loaded ({len(chunks)} chunks)")
    else:
        st.error("❌ Database Load Failed")
        
    st.metric("Queries Processed", len(st.session_state.query_history))
    
    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏥 Patient Case Description")
    
    # Example scenarios
    with st.expander("📋 Example Scenarios (Click to use)"):
        examples = [
            "45-year-old male, ingested pesticide 30 minutes ago, vomiting and confused",
            "Child swallowed bleach, needs immediate first aid",
            "Worker exposed to chemicals, having trouble breathing",
            "Patient unconscious after suspected overdose",
            "Mild skin irritation from cleaning product"
        ]
        for ex in examples:
            if st.button(f"📝 {ex[:50]}..."):
                st.session_state.query = ex
                st.rerun()
    
    # Text input
    query = st.text_area(
        "Describe the poisoning case:",
        value=st.session_state.get("query", ""),
        height=150,
        placeholder="Example: 35-year-old female, ingested 50ml of pesticide 1 hour ago, nausea and sweating..."
    )
    
    analyze = st.button("🚑 Analyze Case", type="primary", use_container_width=True)

with col2:
    st.info("""
    **Include in description:**
    - Age & condition
    - Substance (if known)
    - Amount & time
    - Current symptoms
    """)
    
    st.warning("""
    **🚨 EMERGENCY**
    If unconscious, not breathing, or seizing:
    **Call emergency services NOW!**
    """)

# Process query
if analyze and query:
    if index is None or chunks is None:
        st.error("❌ System is not ready. Document indexing failed. Please check your data folder and PDF.")
    else:
        with st.spinner("Analyzing case..."):
            try:
                start_time = time.time()
                
                # 1. Safety Triage
                safety = safety_triage(query)
                
                # 2. Hybrid Retrieval
                retrieved = hybrid_retrieve(query, index, chunks)
                
                # 3. Generate Response
                final_response = generate_explainable_response(query, retrieved, safety)
                
                end_time = time.time()
                
                # Save to history
                st.session_state.query_history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "query": query[:50] + "...",
                    "risk": safety["risk_level"]
                })
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Triage Results")
                
                # Risk level display
                risk = safety["risk_level"]
                if risk == "CRITICAL":
                    st.error("🚨 **CRITICAL RISK** - Immediate emergency care required!")
                elif risk == "HIGH":
                    st.warning("⚠️ **HIGH RISK** - Urgent medical evaluation needed!")
                else:
                    st.info("📌 **MODERATE RISK** - Monitor and provide guidance")
                
                # Key information
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Risk Level", risk)
                with res_col2:
                    st.metric("Response Time", f"{(end_time - start_time):.1f}s")
                
                # Reason
                st.info(f"**Reason:** {safety['reason']}")
                
                # Detailed analysis
                with st.expander("🔬 Detailed Clinical Analysis", expanded=True):
                    st.write(final_response)
                
                # Recommended action
                st.success(f"**Recommended Action:** {safety['action']}")
                
                # Disclaimer
                st.caption("⚠️ Medical Disclaimer: AI-assisted recommendations only. Always consult healthcare professionals.")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")

elif analyze and not query:
    st.warning("⚠️ Please enter a patient case description.")

# History display
if st.session_state.query_history:
    st.markdown("---")
    st.subheader("📜 Recent Cases")
    for item in reversed(st.session_state.query_history[-5:]):
        risk_icon = "🔴" if item["risk"] == "CRITICAL" else "🟠" if item["risk"] == "HIGH" else "🟡"
        st.text(f"{risk_icon} [{item['time']}] {item['risk']}: {item['query']}")

# Footer
st.markdown("---")
st.markdown("*Powered by Cohere AI | FAISS Vector Database | RAG Architecture*")
