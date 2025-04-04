import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO
from typing import Tuple, List, Dict

# Load YOLO models
damage_model = YOLO("best2.pt")
car_model = YOLO("yolov8m.pt")

# Severity points for damage types
severity_points = {
    "scratch": 1,
    "dent": 2,
    "rust": 2,
    "paint-damage": 2,
    "crack": 2,
}

def number_object_detected(image):
    results = damage_model(image, verbose=False)
    
    dic = results[0].names
    classes = results[0].boxes.cls.cpu().numpy()
    
    class_count = {}
    unique_elements, counts = np.unique(classes, return_counts=True)
    for e, count in zip(unique_elements, counts):
        class_name = dic[int(e)]
        class_count[class_name] = count

    return class_count, results

def car_detection_and_cropping(image):
    r = car_model(image, verbose=False)

    names = r[0].names
    boxes = r[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = set(r[0].boxes.cls.cpu().numpy())
    detected_classes = [names[i] for i in classes]

    if boxes.size != 0 and any(cls in detected_classes for cls in ["car", "truck"]):
        area = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        max_index = np.argmax(area)

        x1, y1, x2, y2 = boxes[max_index]
        cropped_image = image[y1:y2, x1:x2]

        class_count, results = number_object_detected(cropped_image)
        return class_count, results, cropped_image
    else:
        class_count, results = number_object_detected(image)
        return class_count, results, image

def calculate_condition_score(detections):
    total_score = sum(severity_points.get(detection, 1) * count 
                      for detection, count in detections.items() 
                      if detection in severity_points)
    return total_score

def normalize_score(score, max_score=10):
    return min((score / max_score) * 10, 10)

def estimate_condition(detections):
    if not detections:
        return "Excellent"
    
    score = calculate_condition_score(detections)
    normalized_score = normalize_score(score)

    if normalized_score <= 2:
        return "Excellent"
    elif 2 < normalized_score <= 7:
        return "Good"
    elif 7 < normalized_score < 15:
        return "Fair"
    elif 15 <= normalized_score <= 20:
        return "Poor"
    else:
        return "Very Poor"

def process_video(video_path, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return "Error", [], {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    processed_frames = []
    damage_summary = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            damage, results, cropped_frame = car_detection_and_cropping(frame)

            for key, value in damage.items():
                damage_summary[key] = damage_summary.get(key, 0) + value

            for r in results:
                im_array = r.plot(pil=True)
                im_array = np.array(im_array)
                rgb_image = im_array[..., ::-1]
                processed_frames.append(rgb_image)

        frame_count += 1

    cap.release()
    condition = estimate_condition(damage_summary)
    return condition, processed_frames, damage_summary

def create_damage_report(condition, damage_summary, processed_frames):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = f"# 🚗 Comprehensive Car Damage Assessment Report\n"
    report += f"## Report Generated: {timestamp}\n\n"
    report += f"## Overall Vehicle Condition: {condition}\n\n"

    if not damage_summary:
        report += "### 💯 No Damage Detected\n"
        report += "- The vehicle appears to be in pristine condition.\n"
        report += "- No scratches, dents, or other damages were identified during the analysis.\n\n"
    else:
        report += "## 🔍 Detailed Damage Analysis:\n"
        report += "### Damage Types Detected:\n"
        
        for damage_type, count in damage_summary.items():
            severity = severity_points.get(damage_type, 1)
            severity_desc = "Low" if severity <= 1 else "Medium" if severity <= 2 else "High"
            
            report += f"- **{damage_type.capitalize()}**:\n"
            report += f"  - Instances: {count}\n"
            report += f"  - Severity: {severity_desc}\n\n"

    report += "## 🛠️ Repair Recommendations:\n"
    if condition == "Excellent":
        report += "- No repairs needed. Vehicle is in top condition.\n"
    elif condition == "Good":
        report += "- Minor cosmetic touch-ups recommended.\n"
    elif condition == "Fair":
        report += "- Professional assessment and targeted repairs advised.\n"
    elif condition == "Poor":
        report += "- Comprehensive repair plan strongly recommended.\n"
    elif condition == "Very Poor":
        report += "- Extensive repairs or potential vehicle replacement advised.\n"

    selected_frames = processed_frames[:min(5, len(processed_frames))]
    
    return report, selected_frames

def process_data(file):
    # Validate file
    if file is None:
        st.error("No file uploaded.")
        return "Error", None, "Please upload a video file."
    
    # Check file extension
    file_extension = os.path.splitext(file)[1].lower()
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if file_extension not in valid_extensions:
        st.error(f"Invalid file type. Please upload a video file {valid_extensions}")
        return "Error", None, f"Invalid file type. Please upload a video file {valid_extensions}"
    
    try:
        condition, processed_frames, damage_summary = process_video(file)
        report, selected_frames = create_damage_report(condition, damage_summary, processed_frames)
        
        return condition, selected_frames, report
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "Error", None, f"Processing failed: {str(e)}"

def run_car_damage_detection():
    st.header("🚘 360° Car Damage Detection")
    st.write("Upload a video of a car to detect scratches, dents, and other damages.")
    
    uploaded_file = st.file_uploader("Upload 360° Video of Car", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process the video
        condition, selected_frames, report = process_data(tmp_file_path)
        
        # Display condition
        st.subheader(f"Car Condition: {condition}")
        
        # Display damage detection frames
        if selected_frames and len(selected_frames) > 0:
            st.subheader("Damage Detection Samples:")
            cols = st.columns(len(selected_frames))
            for i, frame in enumerate(selected_frames):
                with cols[i]:
                    st.image(frame, caption=f"Frame {i+1}")
        else:
            st.info("No damage detection frames were generated.")
        
        # Display comprehensive report
        st.subheader("Comprehensive Damage Report")
        st.markdown(report)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    run_car_damage_detection()