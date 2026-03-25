import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel

@st.cache_resource
def load_clip_model():
    print("Downloading and caching CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# 2. Call the function
clip_model, clip_processor = load_clip_model()

def embed_image(pil_image):
    inputs = clip_processor(images = pil_image, return_tensors = "pt")

    with torch.no_grad():
        outputs = clip_model.vision_model(**inputs)
        pooled_output = outputs[1]
        image_features = clip_model.visual_projection(pooled_output)
        image_features = image_features / image_features.norm(p = 2, dim = -1, keepdim = True)
        return image_features.squeeze().numpy()
    
def embed_text(text):
    inputs = clip_processor(
        text = text,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = 77
    ) 

    with torch.no_grad():
        outputs = clip_model.text_model(**inputs)
        pooled_output = outputs[1]
        text_features = clip_model.text_projection(pooled_output)
        text_features = text_features / text_features.norm(p = 2, dim = -1, keepdim = True)
        return text_features.squeeze().numpy()    
