 ## Project Description
The Multi-Functional Car Insurance App is a comprehensive solution that integrates multiple AI-powered features to streamline car insurance processes. This application combines four key functionalities:

Car Insurance Assistant: An AI chatbot that provides detailed explanations about car insurance policies, terms, and processes using natural language processing.

Car Damage Detection: A computer vision system that analyzes vehicle videos to detect and assess damage (scratches, dents, rust, etc.) and generates comprehensive damage reports.

Face Recognition: A secure user verification system that registers and authenticates users through facial recognition and number plate matching.

Number Plate Detection: An automated system that extracts vehicle number plates from video footage using OCR technology.

 ## Key Features
AI-Powered Insurance Assistant: Leverages Groq's LLaMA model and RAG pipeline to provide accurate insurance information

Damage Assessment: Uses YOLOv8 models to detect and classify vehicle damage with severity scoring

Biometric Verification: Combines face recognition with number plate matching for secure authentication

Automated Plate Recognition: Implements EasyOCR and custom pattern matching for accurate plate detection

Comprehensive Reporting: Generates detailed damage assessment reports with visual evidence

User Management: Secure database storage for user profiles with facial encodings

 ## Technologies Used
Backend: Python, Streamlit

AI/ML: YOLOv8, OpenCV, EasyOCR, HuggingFace Embeddings

NLP: Groq LLaMA-3, LangChain, RAG pipeline

Database: SQLite

APIs: OCR.space API

Computer Vision: OpenCV, Haarcascades
