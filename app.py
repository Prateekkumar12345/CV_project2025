import streamlit as st
from car_insurance_assistant import run_car_insurance_assistant
from car_damage_detection import run_car_damage_detection
from face_recognition import run_face_recognition
from number_plate_detection import run_number_plate_detection

def main():
    st.title("Multi-Functional Car Insurance App")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Car Insurance Assistant", 
        "Car Damage Detection", 
        "Face Recognition", 
        "Number Plate Detection"
    ])

    if page == "Car Insurance Assistant":
        run_car_insurance_assistant()
    elif page == "Car Damage Detection":
        run_car_damage_detection()
    elif page == "Face Recognition":
        run_face_recognition()
    elif page == "Number Plate Detection":
        run_number_plate_detection()

if __name__ == "__main__":
    main()