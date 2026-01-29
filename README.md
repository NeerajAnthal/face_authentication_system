# 📸 Face Authentication Attendance System

## 🔍 Overview
This project is a real-time **Face Authentication Attendance System** that uses computer vision and deep learning to automatically mark **Punch In** and **Punch Out** using facial recognition.

The system captures live video from a webcam, identifies registered users, and records attendance in a CSV (Excel-compatible) file. Unknown users are detected but ignored for attendance.

---

## 🎯 Features
- Real-time face detection using **MTCNN**
- Face recognition using **FaceNet embeddings**
- Secure user registration (one-time process)
- Automatic **Punch In / Punch Out**
- Prevents duplicate attendance entries
- Handles user absence and re-entry logic
- Attendance stored in CSV format
- Streamlit-based UI
- Works with real camera input

---

## 🧠 System Architecture

The system is divided into **two independent modules**:

### 1️⃣ Face Registration Module (`register.py`)
- Used once per user
- Captures multiple face images
- Generates FaceNet embeddings
- Stores a single **mean embedding per user**

### 2️⃣ Attendance Module (`app.py`)
- Runs daily
- Uses registered embeddings
- Performs real-time face recognition
- Marks Punch In and Punch Out automatically

This separation ensures clean design, security, and scalability.

---

## 🛠️ Technologies Used
- Python 3
- OpenCV
- MTCNN
- FaceNet (keras-facenet)
- NumPy
- Pandas
- Streamlit
- ngrok (for evaluation access)

---


---

## ▶️ How to Run the Project

### 🔹 Step 1: Install Dependencies
```bash
pip install opencv-python mtcnn keras-facenet numpy pandas tensorflow tf-keras streamlit pillow
```

### Step 2: Register a User (One-Time)
```python register.py```
##Look at the camera

##Slightly move your face

##The system captures face images and stores embeddings

###Step 3: Run Attendance System
```streamlit run app.py```
##Click Start Camera

##Registered face → Punch In

##Leave frame → Return → Punch Out


