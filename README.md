# Road Defect Detector via Address

A Streamlit web application that detects road defects using Google Street View images based on user-entered addresses. The app leverages a YOLO object detection model to identify visible road defects and generates downloadable PDF reports.

---

## Features

- Input a road address and fetch corresponding Google Street View images.
- Detect various road defects using a pre-trained YOLO model.
- Display annotated detection results alongside original images.
- Maintain a detection log within the session.
- Generate and download PDF reports summarizing detection results and location info.

---

## Requirements

- Python 3.8+
- Streamlit
- ultralytics (YOLO)
- OpenCV (`cv2`)
- Pillow
- requests
- fpdf
- numpy

Install dependencies with:

```bash
pip install streamlit ultralytics opencv-python pillow requests fpdf numpy

Project Structure
.
├── assets/
│   └── Untitled design.jpg
├── reports/
│   └── (auto-generated PDF reports)
├── roadmaps_app.py
├── best.torchscript
├── README.md


---

Credits
Developed by Sultan-Othman Adekoya
In association with 3MTT DeepTech_Ready and IET
