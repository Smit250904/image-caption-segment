
import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models once
@st.cache_resource
def load_models():
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    caption_model.eval()
    seg_model.eval()
    return caption_processor, caption_model, seg_model

processor, cap_model, seg_model = load_models()

st.title("🧠 Image Captioning & Segmentation")
st.write("Upload an image and get a caption with segmentation overlay.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # === Captioning ===
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = cap_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    st.subheader("📝 Caption")
    st.write(caption)

    # === Segmentation ===
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = seg_model(img_tensor)

    np_image = np.array(image)

    for i in range(min(3, len(prediction[0]['masks']))):
        score = prediction[0]['scores'][i].item()
        if score > 0.5:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            box = prediction[0]['boxes'][i].cpu().numpy()
            rgba_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            np_image = cv2.addWeighted(np_image, 1.0, rgba_mask, 0.5, 0)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(np_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.subheader("🎯 Segmented Image")
    st.image(np_image, use_column_width=True)
