
import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from fpdf import FPDF
from datetime import datetime
import os
import base64
from pathlib import Path

# --- Config ---
st.set_page_config(page_title="Road Defect Detection via Address", layout="centered")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.makedirs("reports", exist_ok=True)

# --- Load model ---
@st.cache_resource
def load_model():
    return YOLO("best.torchscript")  # Change path if needed

model = load_model()

# --- Google Maps helpers ---
def get_coordinates(address):
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_API_KEY}
    response = requests.get(geocode_url, params=params).json()
    if not response["results"]:
        return None
    loc = response["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]

def get_street_view_image_url(lat, lon):
    return (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size=640x640&location={lat},{lon}&fov=80&heading=0&pitch=0&key={GOOGLE_API_KEY}"
    )

def fetch_image_from_url(image_url):
    response = requests.get(image_url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# --- PDF Report Generator ---
def generate_pdf(address, detections, lat=None, lon=None):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Road Defect Detection Report", 0, 1, "C")

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(0, 10, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 10, f"Address: {address}", ln=1)
    if lat and lon:
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
        pdf.cell(0, 10, f"Location: {maps_url}", ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detected Defects:", ln=1)

    pdf.set_font("Arial", size=12)
    if detections:
        for d in detections:
            pdf.cell(0, 10, f"- {d['name']} ({d['confidence']:.2f}%)", ln=1)
    else:
        pdf.cell(0, 10, "No defects detected.", ln=1)

    path = f"reports/defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(path)
    return path

# --- Streamlit App ---
st.title("📍 Road Defect Detector")
st.markdown("Enter an address, and we’ll detect visible road defects from Google Street View.")

address = st.text_input("🛣️ Enter Road Address")
if st.button("Fetch & Detect"):
    if not address:
        st.warning("Please enter an address.")
    else:
        with st.spinner("Fetching location and image..."):
            coords = get_coordinates(address)
            if not coords:
                st.error("Address not found.")
            else:
                lat, lon = coords
                street_img_url = get_street_view_image_url(lat, lon)
                image_cv = fetch_image_from_url(street_img_url)
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                st.image(image_rgb, caption="Street View Image", use_container_width=True)

                with st.spinner("Detecting defects..."):
                    results = model(image_cv, task="detect")


                    # Parse detections
                    detections = []
                    for det in results[0].boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls = det
                        name = model.names[int(cls)]
                        confidence = float(conf * 100)
                        detections.append({"name": name, "confidence": confidence})

                    annotated_image = results[0].plot()
                    st.image(annotated_image, caption="Detection Output", use_container_width=True)

                    if detections:
                        st.markdown("### 🔍 Detected Defects")
                        for item in detections:
                            st.write(f"- {item['name']} ({item['confidence']:.2f}%)")
                    else:
                        st.info("No visible road defects were detected.")

                    # --- PDF Report ---
                    report_path = generate_pdf(address, detections, lat, lon)
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=f,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                            )

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

local_image_path = os.path.join("assets", "Untitled design.jpg")
base64_img = get_base64_image(local_image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)),
            url("data:image/jpeg;base64,{base64_img}");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
# ------- Custom footer with social media links ------- #
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px;
        z-index: 999;
    }
    .footer a {
        color: white;
        text-decoration: none;
        margin: 0 10px;
    }
    </style>
    <div class="footer">
        Developed by Sultan-Othman Adekoya - ©2025 | In Association with 3MTT DeepTech_Ready IET
        <br>
        <a href="https://www.instagram.com/tobisultanothman/" target="_blank">Instagram</a>
        <a href="https://www.linkedin.com/in/sultan-othman/" target="_blank">LinkedIn</a>
        <a href="https://github.com/Sultan-Othman-Adekoya" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .stApp {{
        background: 
            linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), 
            url("data:image/jpeg;base64,{base64_img}");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# ------- Custom footer with social media links ------- #
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px;
        z-index: 999;
    }
    .footer a {
        color: white;
        text-decoration: none;
        margin: 0 10px;
    }
    </style>
    <div class="footer">
        Developed by Sultan-Othman Adekoya - ©2025 | In Association with 3MTT DeepTech_Ready IET
        <br>
        <a href="https://www.instagram.com/tobisultanothman/" target="_blank">Instagram</a>
        <a href="https://www.linkedin.com/in/sultan-othman/" target="_blank">LinkedIn</a>
        <a href="https://github.com/Sultan-Othman-Adekoya" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

          
