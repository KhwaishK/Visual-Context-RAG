# 🖼️ Visual Context RAG

[![Live Demo on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/KhwaishK/Visual-Context-RAG)

**Visual Context RAG** is an end-to-end Multimodal Retrieval-Augmented Generation pipeline. It allows users to upload complex, unstructured PDFs and query both the textual data and the embedded visual data (charts, graphs, and diagrams) simultaneously.

By projecting text and spatial image tensors into a shared, unified vector space, this application bypasses the traditional limitations of text-only RAG systems.

---

*(Drag and drop your screenshot of the app working right here while editing in GitHub! It will replace this text with the image code.)*

---

## ⚙️ Architecture & Data Flow
1. **Document Ingestion:** Uses `PyMuPDF` (fitz) to parse PDFs, extracting standard text chunks and ripping embedded images/charts as raw byte streams.
2. **Unified Embedding Space:** Utilizes OpenAI's `CLIP-ViT-Base-Patch32` to generate 512-dimensional embeddings for *both* text chunks and images, mapping them into the same semantic vector space.
3. **Vector Storage:** Stores the multimodal embeddings in a localized, CPU-optimized `FAISS` index for lightning-fast similarity search.
4. **Context Retrieval & Generation:** Queries are embedded via CLIP, compared against the FAISS index, and the top-K multimodal results (text and base64 images) are injected into the context window of Meta's `Llama 4 Scout` model via the Groq API.

## 🛠️ Tech Stack
* **Frontend/Deployment:** Streamlit
* **Deep Learning Framework:** PyTorch
* **Multimodal Embeddings:** Hugging Face `transformers` (OpenAI CLIP)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **LLM Engine:** Groq API (Llama 4 Scout)
* **Document Processing:** PyMuPDF (`fitz`), Pillow

## 🔬 Architectural Limitations & Future Work
This project was built to test the limits of unified vision-language vector spaces. Current known limitations include:
* **The 77-Token Limit:** CLIP is optimized for short image captions. Dense text paragraphs exceeding ~60 words suffer from truncation. *Future iteration: Implement a decoupled dual-encoder architecture (routing text through `SentenceTransformers` and images through `CLIP`).*
* **Spatial Table Loss:** Flattening tabular data into text strings strips spatial bounding-box coordinates, occasionally leading to LLM hallucination on financial tables. *Future iteration: Integrate `Unstructured.io` or `TableTransformer` for precise grid preservation.*

## 💻 Run it Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KhwaishK/Visual-Context-RAG.git](https://github.com/KhwaishK/Visual-Context-RAG.git)
   cd Visual-Context-RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment:**
   * Create a ```.env``` file in the root directory and add your Groq API key: *
   ```bash
   GROQ_API_KEY= "your_api_key_here"

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   
