# Multi-Functional Car Insurance App

## Project Description

The **Multi-Functional Car Insurance App** is a comprehensive AI-powered solution that streamlines various aspects of car insurance. The application integrates multiple intelligent features including:

- **Car Insurance Assistant**: An AI chatbot that answers queries related to car insurance policies, processes, and terminology using NLP.
- **Car Damage Detection**: A computer vision module that analyzes vehicle videos to detect and assess visible damages (scratches, dents, rust, etc.).
- **Face Recognition**: A biometric verification system for registering and authenticating users via facial recognition combined with number plate matching.
- **Number Plate Detection**: Automatically extracts and identifies vehicle number plates from video footage using OCR.

---

## Key Features

- ✅ **AI-Powered Insurance Assistant**: Utilizes Groq's LLaMA model and a RAG pipeline to provide accurate insurance-related information.
- ✅ **Damage Assessment System**: Employs YOLOv8 to detect, localize, and classify different types of car damage with severity scores.
- ✅ **Biometric User Authentication**: Combines face recognition and number plate matching for secure user verification.
- ✅ **Number Plate Recognition**: Uses EasyOCR and custom regex pattern matching for precise license plate detection.
- ✅ **Comprehensive Reporting**: Generates damage reports with annotated visual evidence.
- ✅ **User Management**: Secure storage of user profiles and facial encodings using SQLite.

---

## Technologies Used

- **Backend**: Python, Streamlit  
- **AI/ML Models**: YOLOv8, OpenCV, Haarcascades, EasyOCR  
- **NLP & LLMs**: Groq LLaMA-3, LangChain, RAG pipeline  
- **Database**: SQLite  
- **APIs**: OCR.space API  
- **Computer Vision**: OpenCV, YOLOv8  

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/car-insurance-app.git
cd car-insurance-app
