import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from embedding_functions import embed_text

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

def retrieve_multimodal(query, vector_store, k=5):
    query_embedding = embed_text(query)
    results = vector_store.similarity_search_by_vector(
        embedding = query_embedding,
        k = k
    )
    return results

def create_multimodal_message(query, retrieved_docs, image_data_store):
    content = []

    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })

    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })

    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })  
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })  
 
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })        

    return HumanMessage(content = content)

def answer_query(query, vector_store, image_data_store):
    context_docs = retrieve_multimodal(query, vector_store, k=5)
    message = create_multimodal_message(query, context_docs, image_data_store)
    response = llm.invoke([message])
    return response.content