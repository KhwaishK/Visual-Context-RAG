import fitz
import io
import base64
import numpy as np
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embedding_functions import embed_text, embed_image

def process_uploaded_pdf(pdf_bytes):
    
    doc = fitz.open(stream = pdf_bytes, filetype = "pdf")

    all_docs = []
    all_embeddings = []
    image_data_store = {}

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)

    for i,page in enumerate(doc): 
        # TEXT PROCESSING
        text = page.get_text() 
        if text.strip():
            temp_doc = Document(page_content = text, metadata = {"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content)
                all_embeddings.append(embedding)
                all_docs.append(chunk)

        # IMAGE PROCESSING
        for img_index, img in enumerate(page.get_images(full = True)):
            try:
                xref = img[0] 
                base_image = doc.extract_image(xref) 
                image_bytes = base_image["image"] 

                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Create unique identifier 
                image_id = f"page_{i}_img_{img_index}"

                buffered = io.BytesIO()
                pil_image.save(buffered, format = "PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                # Embed image using CLIP
                embedding = embed_image(pil_image)
                all_embeddings.append(embedding)

                # Create document for image
                image_doc = Document(
                    page_content = f"[Image: {image_id}]",
                    metadata = {"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)

            except Exception as e: 
                print(f"Error processing image {img_index} on page {i}: {e}")
                continue

    doc.close()

    # BUILD FAISS DATABASE
    embeddings_array = np.array(all_embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings = [(d.page_content, emb) for d, emb in zip(all_docs, embeddings_array)],
        embedding = None,
        metadatas = [d.metadata for d in all_docs]
    )
    
    return vector_store, image_data_store

