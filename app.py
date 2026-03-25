import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

from pdf_processor import process_uploaded_pdf
from rag_pipeline import answer_query

st.set_page_config(page_title="Enterprise Multimodal RAG", page_icon="🧠", layout="wide")

st.title("📚 Enterprise Multimodal RAG Pipeline")
st.markdown("""
Upload a complex PDF (containing both text and images). This system uses a custom PyTorch/CLIP pipeline to project both modalities into a shared vector space, preventing duplicate ingestion, and answers questions using Meta's Llama 4 Scout.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "image_data_store" not in st.session_state:
    st.session_state.image_data_store = None

with st.sidebar:
    st.header("⚙️ Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Extracting tensors and building FAISS index..."):

            pdf_bytes = uploaded_file.read()
   
            v_store, img_store = process_uploaded_pdf(pdf_bytes)
          
            st.session_state.vector_store = v_store
            st.session_state.image_data_store = img_store
            
            st.success("✅ Document Successfully Indexed!")
            st.info("""
            🚀 **Smart Ingestion Triggered:**
            - Scanned document for new multi-modal embeddings.
            - Incremental indexing applied to prevent vector duplication.
            - Ready for queries.
            """)

# --- MAIN CHAT INTERFACE ---
if st.session_state.vector_store is not None:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF's text or charts..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching shared vector space and analyzing visual data..."):
                response = answer_query(
                    query=prompt, 
                    vector_store=st.session_state.vector_store, 
                    image_data_store=st.session_state.image_data_store
                )
                st.markdown(response)
   
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 Please upload and process a PDF in the sidebar to start chatting.")