import streamlit as st
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
from mtcnn import MTCNN
import time
import os
from numpy.linalg import norm
from keras_facenet import FaceNet

# ================= STREAMLIT CONFIG =================
st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("📸 Face Authentication Attendance System")

# ================= CONFIG =================
attendance_file = "attendance.csv"
FACE_ABSENCE_TIME = 8        # seconds
DISPLAY_TIME = 3
THRESHOLD = 0.35

# ================= FILE SETUP =================
if not os.path.exists(attendance_file):
    pd.DataFrame(
        columns=["User_ID", "Date", "Punch_In", "Punch_Out"]
    ).to_csv(attendance_file, index=False)

# ================= LOAD MODELS =================
detector = MTCNN()
embedder = FaceNet()

# ================= LOAD REGISTERED EMBEDDINGS (FIX APPLIED) =================
embedding_db = {}

if os.path.exists("data/embeddings"):
    for file in os.listdir("data/embeddings"):
        if file.endswith(".npy"):
            user_id = file.replace(".npy", "")
            emb = np.load(f"data/embeddings/{file}")

            # 🔴 FIX OPTION 2: convert (N,512) → (512,)
            if len(emb.shape) == 2:
                emb = np.mean(emb, axis=0)

            embedding_db[user_id] = emb

st.sidebar.success(f"Registered users: {list(embedding_db.keys())}")

# ================= COSINE DISTANCE =================
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

# ================= ATTENDANCE LOGIC =================
def mark_attendance(user_id):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)
    user_today = df[(df["User_ID"] == user_id) & (df["Date"] == today)]

    if user_today.empty:
        new_row = {
            "User_ID": user_id,
            "Date": today,
            "Punch_In": time_now,
            "Punch_Out": ""
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        action = "Punch In Successful"
    else:
        idx = user_today.index[0]
        punch_out_val = df.loc[idx, "Punch_Out"]

        if pd.isna(punch_out_val) or str(punch_out_val).strip() == "":
            df.loc[idx, "Punch_Out"] = time_now
            action = "Punch Out Successful"
        else:
            action = "Already Completed Today"

    df.to_csv(attendance_file, index=False)
    return action

# ================= STATE =================
last_seen = {}
user_present = {}
display_info = {"text": "", "color": (0, 255, 0), "until": 0}

# ================= STREAMLIT CAMERA =================
run = st.checkbox("▶ Start Camera")
frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(rgb)
    except:
        faces = []

    current_time = time.time()
    detected_users = []

    for face in faces:
        if face["confidence"] > 0.95:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            face_crop = rgb[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            # ---------- FACE EMBEDDING ----------
            face_img = cv2.resize(face_crop, (160, 160))
            face_img = np.expand_dims(face_img, axis=0)
            test_embedding = embedder.embeddings(face_img)[0]

            # ---------- FACE MATCHING ----------
            identity = "Unknown"
            min_dist = float("inf")

            for user_id, ref_embedding in embedding_db.items():
                dist = float(cosine_distance(test_embedding, ref_embedding))
                if dist < min_dist:
                    min_dist = dist
                    identity = user_id

            if min_dist > THRESHOLD:
                identity = "Unknown"

            # ---------- ATTENDANCE ----------
            if identity != "Unknown":
                detected_users.append(identity)
                last_seen[identity] = current_time

                if not user_present.get(identity, False):
                    action_msg = mark_attendance(identity)
                    display_info = {
                        "text": f"{identity}: {action_msg}",
                        "color": (0, 255, 0) if "Successful" in action_msg else (255, 165, 0),
                        "until": current_time + DISPLAY_TIME
                    }
                    user_present[identity] = True

                if current_time < display_info["until"]:
                    label = display_info["text"]
                    color = display_info["color"]
                else:
                    label = f"{identity} (Present)"
                    color = (255, 255, 0)
            else:
                label = "Unknown"
                color = (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ---------- ABSENCE CHECK ----------
    for uid in list(user_present.keys()):
        if uid not in detected_users:
            if (current_time - last_seen.get(uid, 0)) > FACE_ABSENCE_TIME:
                user_present[uid] = False

    frame_window.image(frame)

cap.release()
