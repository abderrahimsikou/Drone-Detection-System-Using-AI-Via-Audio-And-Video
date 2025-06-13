import os
import librosa
import numpy as np
import cv2
import streamlit as st
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide", page_title="Professional Drone Detection")
st.markdown("""
<style>
    .reportview-container {background: #f0f2f6}
    .metric {box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 15px; border-radius: 10px}
    .alert {font-size: 18px !important; font-weight: bold !important}
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    audio_model = load_model("models/drone_audio_model.h5")
    video_model = YOLO("models/drone_YOLO_model.pt")
    return audio_model, video_model

audio_model, video_model = load_models()

def process_file(uploaded_file):
    with open("temp_media.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_media.mp4"

def analyze_media(file_path):
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎬 Media Preview")
        if file_path.endswith(('.mp4', '.avi')):
            st.video(file_path)
        else:
            st.image(file_path)
    
    with col2:
        st.header("🔍 Analysis Results")
        
        # Analyze Video
        with st.spinner("Analyzing visual content..."):
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                results = video_model.predict(frame)
                annotated_frame = results[0].plot()
                st.image(annotated_frame, caption="Detection Results", use_column_width=True)
                video_detected = any(results[0].names[int(cls)].lower() == 'drone' 
                                   for cls in results[0].boxes.cls.cpu().numpy())
            else:
                video_detected = False
        
        # Analyze Audio
        audio_detected = False
        if file_path.endswith(('.mp4', '.avi')):
            with st.spinner("Analyzing audio..."):
                audio_path = "extracted_audio.wav"
                if extract_audio(file_path, audio_path):
                    audio_confidence = analyze_audio(audio_path, audio_model)
                    audio_detected = audio_confidence > 0.5
                    st.progress(audio_confidence)
                    st.metric("Audio Confidence", f"{audio_confidence:.2%}")
        
        # Show results
        if video_detected and audio_detected:
            st.error("🚨 ALERT: Drone Confirmed (Audio + Visual)!")
        elif video_detected:
            st.warning("👁️ Drone Detected (Visual Only)")
        elif audio_detected:
            st.warning("🔊 Drone Detected (Audio Only)")
        else:
            st.success("✅ No Drone Detected")

# Main Interface
st.title("🚁 Professional Drone Detection System")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Video/Image", type=['mp4', 'avi', 'jpg', 'png'])
if uploaded_file:
    file_path = process_file(uploaded_file)
    analyze_media(file_path)
    os.remove(file_path) 