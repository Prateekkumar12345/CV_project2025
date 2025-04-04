import streamlit as st
import sqlite3
import numpy as np
import cv2
import tempfile
import easyocr
import re
import time
import requests
from PIL import Image
import io

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def create_database():
    """Create SQLite database for storing user information"""
    conn = sqlite3.connect("face_database.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone_number TEXT,
            number_plate TEXT,
            face_encoding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def detect_and_encode_face(image):
    """
    Detect and encode face using OpenCV
    
    Args:
        image: numpy array or PIL Image
    
    Returns:
        Face encoding or None if no face detected
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        st.warning("No face detected in the image!")
        return None
    
    # Take the first detected face
    (x, y, w, h) = faces[0]
    
    # Extract the face region
    face_img = gray[y:y+h, x:x+w]
    
    # Resize face to a standard size
    face_img = cv2.resize(face_img, (100, 100))
    
    # Convert to blob for storage
    face_encoding = face_img.tobytes()
    
    return face_encoding

def register_user(image, name, phone_number, number_plate):
    """
    Register a new user in the database
    
    Args:
        image: Image to use for face encoding
        name: User's name
        phone_number: User's phone number
        number_plate: User's vehicle number plate
    
    Returns:
        Boolean indicating successful registration
    """
    try:
        # Validate inputs
        if not all([name, phone_number, number_plate, image]):
            st.error("Please fill in all fields and provide an image")
            return False

        # Detect and encode face
        face_encoding = detect_and_encode_face(image)
        
        if face_encoding is None:
            st.error("Face detection failed. Please use a clear image with a visible face.")
            return False
        
        # Database connection
        conn = sqlite3.connect("face_database.db")
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT * FROM users WHERE name = ? OR phone_number = ? OR number_plate = ?", 
                       (name, phone_number, number_plate))
        existing_user = cursor.fetchone()
        
        if existing_user:
            st.warning("A user with similar details already exists!")
            conn.close()
            return False
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (name, phone_number, number_plate, face_encoding) VALUES (?, ?, ?, ?)",
            (name, phone_number, number_plate, face_encoding)
        )
        conn.commit()
        conn.close()
        
        st.success(f"User {name} registered successfully!")
        return True
    
    except Exception as e:
        st.error(f"Error registering user: {e}")
        return False

def verify_user(image, number_plate=None):
    """
    Verify a user by face and optionally number plate
    
    Args:
        image: Image to verify
        number_plate: Optional number plate for additional verification
    
    Returns:
        Verified user details or None
    """
    try:
        # Detect face in captured image
        captured_face_encoding = detect_and_encode_face(image)
        
        if captured_face_encoding is None:
            return None
        
        # Database connection
        conn = sqlite3.connect("face_database.db")
        cursor = conn.cursor()
        
        # Fetch all registered users
        if number_plate:
            # If number plate is provided, filter by number plate
            cursor.execute(
                "SELECT name, phone_number, number_plate, face_encoding FROM users WHERE number_plate = ?", 
                (number_plate,)
            )
        else:
            # Otherwise, fetch all users
            cursor.execute("SELECT name, phone_number, number_plate, face_encoding FROM users")
        
        users = cursor.fetchall()
        conn.close()
        
        # Check for matching users
        for name, phone_number, stored_plate, stored_face_encoding in users:
            # Compare face encodings (simple bytewise comparison)
            face_matches = (captured_face_encoding == stored_face_encoding)
            
            # If face matches (and optionally number plate matches)
            if face_matches and (not number_plate or number_plate.lower() == stored_plate.lower()):
                st.success(f"User Verified: {name}")
                return {
                    "name": name, 
                    "phone_number": phone_number, 
                    "number_plate": stored_plate
                }
        
        st.error("No matching user found")
        return None
    
    except Exception as e:
        st.error(f"Verification error: {e}")
        return None

def real_time_capture():
    """
    Capture image from webcam
    
    Returns:
        Captured image as numpy array or None
    """
    # Open the webcam
    try:
        cap = cv2.VideoCapture(0)
        
        # Check if webcam is opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open webcam. Please check your camera connection.")
            return None
        
        # Attempt to read multiple frames to ensure a good capture
        for _ in range(5):
            ret, frame = cap.read()
        
        # Release the webcam
        cap.release()
        
        if not ret:
            st.error("Failed to capture frame. Please try again.")
            return None
        
        return frame
    
    except Exception as e:
        st.error(f"Webcam capture error: {e}")
        return None

def run_face_recognition():
    """
    Run the face recognition application
    """
    create_database()
    
    st.title("Face Recognition System")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Register", "Verify", "Real-Time Capture"])
    
    with tab1:
        st.header("User Registration")
        
        # Registration inputs
        name = st.text_input("Name", key="reg_name")
        phone_number = st.text_input("Phone Number", key="reg_phone")
        number_plate = st.text_input("Number Plate", key="reg_plate")
        
        # Image upload
        reg_image = st.file_uploader("Upload Registration Image", 
                                     type=["jpg", "png", "jpeg"], 
                                     key="reg_image")
        
        if reg_image:
            # Convert uploaded file to PIL Image
            pil_image = Image.open(reg_image)
            st.image(pil_image, caption="Uploaded Image", use_column_width=True)
        
        # Register button
        if st.button("Register User"):
            if reg_image and name and phone_number and number_plate:
                # Try registering the user
                success = register_user(pil_image, name, phone_number, number_plate)
                if success:
                    st.rerun()  # Replace experimental_rerun with rerun
    
    with tab2:
        st.header("User Verification")
        
        # Verification image upload
        verify_image = st.file_uploader("Upload Verification Image", 
                                        type=["jpg", "png", "jpeg"], 
                                        key="verify_image")
        
        # Optional number plate input
        verify_plate = st.text_input("Optional Number Plate", key="verify_plate")
        
        if verify_image:
            # Convert uploaded file to PIL Image
            pil_image = Image.open(verify_image)
            st.image(pil_image, caption="Uploaded Image", use_column_width=True)
        
        # Verify button
        if st.button("Verify User"):
            if verify_image:
                # Verify user with optional number plate
                result = verify_user(pil_image, verify_plate)
                if result:
                    st.write("Verified User Details:")
                    st.write(f"Name: {result['name']}")
                    st.write(f"Phone Number: {result['phone_number']}")
                    st.write(f"Number Plate: {result['number_plate']}")
    
    with tab3:
        st.header("Real-Time Capture")
        
        # Capture button
        if st.button("Capture from Webcam"):
            captured_frame = real_time_capture()
            
            if captured_frame is not None:
                # Convert OpenCV image (BGR) to PIL image (RGB)
                captured_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                st.image(captured_image, caption="Captured Image", use_column_width=True)
                
                # Optional verification
                if st.button("Verify Captured Image"):
                    result = verify_user(captured_image)
                    if result:
                        st.write("Verified User Details:")
                        st.write(f"Name: {result['name']}")
                        st.write(f"Phone Number: {result['phone_number']}")
                        st.write(f"Number Plate: {result['number_plate']}")

if __name__ == "__main__":
    run_face_recognition()