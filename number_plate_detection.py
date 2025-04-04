import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import easyocr
import re
import requests

def detect_number_plate(video_path):
    """
    Detect and extract number plate from video
    
    Args:
        video_path: Path to the uploaded video file
    
    Returns:
        Tuple of (detected plate text, detected plate image)
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Dictionary to replace digits with visually similar letters
    digit_to_letter = {"0": "O", "1": "I", "5": "S"}
    
    try:
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Check if video is opened successfully
        if not cap.isOpened():
            st.error("Unable to open video file. Please check the file.")
            return None, None
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("No frames could be read from the video.")
                break
            
            # Image preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
            
            # Fix the typo in thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Detect text using EasyOCR
            results = reader.readtext(thresh)
            
            for (bbox, text, prob) in results:
                # Filter low probability results
                if prob < 0.5:
                    continue
                
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                
                # Draw a rectangle around the detected plate
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                
                # Crop number plate region
                detected_plate_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                
                # Rotate the detected plate image 90 degrees clockwise
                rotated_plate_image = cv2.rotate(detected_plate_image, cv2.ROTATE_90_CLOCKWISE)
                
                # Save the extracted number plate image
                cv2.imwrite('number_plate_image.jpg', rotated_plate_image)
                
                # OCR API processing
                try:
                    api_url = "https://api.ocr.space/parse/image"
                    with open('number_plate_image.jpg', "rb") as img:
                        response = requests.post(
                            api_url, 
                            files={"image": img}, 
                            data={
                                "apikey": "helloworld", 
                                "language": "eng",
                                "isOverlayRequired": False,
                                "filetype": "JPG"
                            },
                            timeout=10
                        )
                    
                    # Check response
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get("ParsedResults"):
                            plate_text = result["ParsedResults"][0]["ParsedText"].strip()
                            
                            # Multiple plate number patterns to match
                            patterns = [
                                r"\b[A-Z]\s*\d{3}\s*[A-Z]{2}\s*\d{3}\b",
                                r"\b\d{2}\s*[A-Z]{2}\s*\d{4}\b",
                                r"\b[A-Z]{1,2}\s*\d{3,4}\s*[A-Z]{2}\s*\d{3}\b"
                            ]
                            
                            # First attempt to match the correct pattern
                            matched_plate = None
                            for pattern in patterns:
                                match = re.search(pattern, plate_text)
                                if match:
                                    matched_plate = match.group().replace(" ", "")
                                    break
                            
                            # Fallback: try with digit-to-letter correction
                            if not matched_plate:
                                corrected_text = "".join(digit_to_letter.get(c, c) for c in plate_text)
                                for pattern in patterns:
                                    match = re.search(pattern, corrected_text)
                                    if match:
                                        matched_plate = match.group().replace(" ", "")
                                        break
                            
                            if matched_plate:
                                cap.release()
                                return matched_plate, rotated_plate_image
                
                except Exception as e:
                    st.error(f"OCR processing error: {e}")
            
            # Process only first frame
            break
        
        cap.release()
        st.warning("Could not detect a valid number plate.")
        return None, None
    
    except Exception as e:
        st.error(f"Error in number plate detection: {e}")
        return None, None

def run_number_plate_detection():
    """
    Streamlit app for number plate detection
    """
    st.header("Number Plate Detection")
    
    # Video file upload
    uploaded_file = st.file_uploader(
        "Upload a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'], 
        help="Please upload a video containing a vehicle with a visible number plate"
    )
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
        
        try:
            # Detect number plate
            st.info("Processing video... Please wait.")
            plate_number, plate_image = detect_number_plate(temp_video_path)
            
            # Display results
            if plate_number and plate_image is not None:
                st.success(f"Detected Number Plate: {plate_number}")
                
                # Display the detected plate image
                st.subheader("Detected Number Plate Image")
                st.image(plate_image, channels="BGR", use_column_width=True)
            else:
                st.warning("No number plate detected in the video.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary files
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            if os.path.exists('number_plate_image.jpg'):
                os.unlink('number_plate_image.jpg')

# This allows the script to be run directly or imported
if __name__ == "__main__":
    run_number_plate_detection()