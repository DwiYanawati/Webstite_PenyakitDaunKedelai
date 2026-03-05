import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Penyakit Daun Kedelai", page_icon="🌿")
st.title("🌿 Deteksi Penyakit Daun Kedelai")

# Load model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

model = load_model()

if model is not None:
    st.success("✅ Model siap digunakan!")
    
    uploaded_file = st.file_uploader("Upload gambar daun kedelai", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Konversi RGB ke BGR untuk OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Deteksi
        with st.spinner("Mendeteksi..."):
            results = model(img_bgr)
            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Tampilkan hasil
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Hasil Deteksi")
            st.image(result_img_rgb, use_container_width=True)
        
        # Info deteksi
        if len(results[0].boxes) > 0:
            st.subheader("📊 Hasil:")
            for i, box in enumerate(results[0].boxes):
                class_name = results[0].names[int(box.cls)]
                conf = float(box.conf)
                st.write(f"{i+1}. {class_name} ({(conf*100):.1f}%)")
        else:
            st.write("Tidak ada penyakit terdeteksi")

else:
    st.error("❌ Model tidak bisa dimuat. Pastikan file best.pt ada di folder yang benar.")
