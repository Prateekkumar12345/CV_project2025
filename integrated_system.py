import streamlit as st
import sqlite3
import numpy as np
import cv2
import tempfile
import easyocr
import re
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

def detect_number_plate(video_path):
    """
    Detect and extract number plate from video
    
    Args:
        video_path: Path to the uploaded video file
    
    Returns:
        Extracted number plate as string or None
    """
    cap = cv2.VideoCapture(video_path)
    digit_to_letter = {"0": "O", "1": "I", "5": "S"}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        results = reader.readtext(thresh)
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            detected_plate_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # Attempt to get license plate text
            cv2.imwrite('number_plate_image.jpg', detected_plate_image)
            image_path = "number_plate_image.jpg"
            
            api_url = "https://api.ocr.space/parse/image"
            
            with open(image_path, "rb") as img:
                response = requests.post(api_url, files={"image": img}, data={"apikey": "helloworld", "language": "eng"})
            
            result = response.json()
            
            if result["ParsedResults"]:
                plate_text = result["ParsedResults"][0]["ParsedText"].strip()
                match = re.search(r"\b[A-Z]\s*\d{3}\s*[A-Z]{2}\s*\d{3}\b", plate_text)
                
                if not match:
                    corrected_text = "".join(digit_to_letter.get(c, c) for c in plate_text)
                    match = re.search(r"\b[A-Z]\s*\d{3}\s*[A-Z]{2}\s*\d{3}\b", corrected_text)
                
                plate_text_final = match.group().replace(" ", "") if match else None
                cap.release()
                return plate_text_final, detected_plate_image
        
        break
    
    cap.release()
    return None, None

def run_integrated_system():
    """
    Main Streamlit application for integrated face and number plate recognition
    """
    create_database()
    st.header("Integrated Face and Number Plate Recognition")
    
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Register", "Verify"])
    
    with tab1:
        st.subheader("User Registration")
        
        # User input
        name = st.text_input("Name", key="reg_name")
        phone_number = st.text_input("Phone Number", key="reg_phone")
        number_plate = st.text_input("Number Plate", key="reg_plate")
        
        # Registration method
        reg_method = st.radio("Choose Registration Method", 
                              ["Upload Image", "Capture Photo"], 
                              key="reg_method")
        
        # Registration image
        reg_image = None
        if reg_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", 
                                             type=["jpg", "png", "jpeg"], 
                                             key="reg_upload")
            if uploaded_file:
                reg_image = Image.open(uploaded_file)
                st.image(reg_image, caption="Uploaded Image", use_container_width=True)
        else:
            st.write("Use OpenCV capture functionality if needed")
        
        # Validate inputs and register
        if st.button("Register User", key="register_btn"):
            # Input validation
            input_errors = []
            
            if not name or len(name.strip()) < 2:
                input_errors.append("Name must be at least 2 characters long")
            
            if not phone_number or not re.match(r'^[0-9]{10}$', phone_number):
                input_errors.append("Phone number must be 10 digits")
            
            if not number_plate or not re.match(r'^[A-Z]{1,2}\d{3,4}[A-Z]{2}\d{3}$', number_plate):
                input_errors.append("Number plate must be in format: P947OT777")
            
            if reg_image is None:
                input_errors.append("An image is required for registration")
            
            # Display or process errors
            if input_errors:
                for error in input_errors:
                    st.error(error)
            else:
                # Attempt registration
                result = register_user(reg_image, name, phone_number, number_plate)
                if result:
                    st.experimental_rerun()
    
    with tab2:
        st.subheader("User Verification")
        
        # Video upload for number plate extraction
        uploaded_video = st.file_uploader("Upload Video for Number Plate", 
                                          type=["mp4", "avi", "mov", "mkv"], 
                                          key="video_upload")
        
        if uploaded_video:
            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract number plate
            detected_plate, plate_image = detect_number_plate(tmp_file_path)
            
            if detected_plate:
                st.success(f"Detected Number Plate: {detected_plate}")
                
                # Image verification
                st.subheader("Face Verification")
                verify_method = st.radio("Choose Verification Method", 
                                         ["Upload Image", "Capture Photo"], 
                                         key="verify_method")
                
                verify_image = None
                if verify_method == "Upload Image":
                    uploaded_file = st.file_uploader("Choose an image to verify", 
                                                     type=["jpg", "png", "jpeg"], 
                                                     key="verify_upload")
                    if uploaded_file:
                        verify_image = Image.open(uploaded_file)
                        st.image(verify_image, caption="Uploaded Image", use_container_width=True)
                
                # Verify button
                if st.button("Verify User", key="verify_btn"):
                    if verify_image:
                        # Perform verification with detected number plate
                        verified_user = verify_user(verify_image, detected_plate)
                        
                        if verified_user:
                            st.write("Verified User Details:")
                            st.write(f"Name: {verified_user['name']}")
                            st.write(f"Phone Number: {verified_user['phone_number']}")
                            st.write(f"Number Plate: {verified_user['number_plate']}")
                    else:
                        st.warning("Please provide an image for verification")

# Main execution
if __name__ == "__main__":
    run_integrated_system()