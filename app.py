import io
import os
from datetime import datetime

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Streamlit page config
st.set_page_config(page_title="License Plate OCR", layout="centered")

# Load YOLO model (ensure the model file exists at the specified path)
MODEL_PATH = "./best.pt"
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# App title
st.title("License Plate Detection and OCR")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Helper to convert Streamlit uploaded file to OpenCV image
def read_image_to_cv(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

# Helper to convert cv2 image to PIL
def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# Helper to annotate image with the first license plate bbox
def annotate_first_plate(img_cv: np.ndarray, results):
    annotated = img_cv.copy()
    plate_bbox = None
    plate_conf = None
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        # Select the highest confidence box
        confidences = r.boxes.conf.cpu().numpy()
        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        idx = int(np.argmax(confidences))
        plate_bbox = boxes_xyxy[idx]  # [x1, y1, x2, y2]
        plate_conf = float(confidences[idx])
        break

    if plate_bbox is not None:
        x1, y1, x2, y2 = plate_bbox
        # Draw rectangle and label
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"Plate: {plate_conf:.2f}"
        ((tw, th), baseline) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return annotated, plate_bbox

# OCR on cropped license plate
def ocr_text_from_plate(img_cv: np.ndarray, bbox) -> str:
    x1, y1, x2, y2 = bbox
    # Crop to bounding box
    plate_crop = img_cv[y1:y2, x1:x2]

    # Preprocess: grayscale, resize, threshold
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(thresh)

    # Tesseract config tuned for license plates
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(pil_img, config=config)

    return text

# Main logic
annotated_img_cv = None
extracted_text = None

if uploaded_file is not None:
    # Read and show original image
    img_cv = read_image_to_cv(uploaded_file)
    st.subheader("Original image")
    st.image(cv_to_pil(img_cv))

    # Run YOLO detection
    with st.spinner("Detecting license plate..."):
        results = model(img_cv, verbose=False)

    # Annotate image
    annotated_img_cv, bbox = annotate_first_plate(img_cv, results)

    if bbox is None:
        st.error("No license plate detected.")
    else:
        st.subheader("Image with bounding box")
        st.image(cv_to_pil(annotated_img_cv))

        # OCR on cropped plate
        with st.spinner("Extracting text with OCR..."):
            extracted_text = ocr_text_from_plate(img_cv, bbox)

        st.subheader("Extracted text")
        st.code(extracted_text if extracted_text else "(No text detected)")

        # Prepare downloads
        pil_annotated = cv_to_pil(annotated_img_cv)
        img_bytes_io = io.BytesIO()
        pil_annotated.save(img_bytes_io, format="PNG")
        img_bytes = img_bytes_io.getvalue()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_filename = f"plate_text_{ts}.txt"
        img_filename = f"plate_annotated_{ts}.png"
        text_bytes = (extracted_text or "").encode("utf-8")

        st.download_button(
            label="Download extracted text (.txt)",
            data=text_bytes,
            file_name=text_filename,
            mime="text/plain",
        )

        st.download_button(
            label="Download license plate image with bounding box (.png)",
            data=img_bytes,
            file_name=img_filename,
            mime="image/png",
        )
