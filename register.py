import cv2
import numpy as np
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet

# ================= CONFIG =================
REQUIRED_IMAGES = 20          
FACE_CONFIDENCE = 0.95

# ================= PATHS =================
FACE_DIR = "data/registered_faces"
EMB_DIR = "data/embeddings"

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# ================= INPUT =================
user_id = input("Enter User ID (unique): ").strip()

embedding_path = os.path.join(EMB_DIR, f"{user_id}.npy")
if os.path.exists(embedding_path):
    print("❌ User already registered!")
    exit()

print(f"\n📸 Registering User: {user_id}")
print(f"Please look at the camera and move slightly.")
print("Press Q to cancel.\n")

# ================= LOAD MODELS =================
detector = MTCNN()
embedder = FaceNet()

# ================= CAPTURE FACES =================
cap = cv2.VideoCapture(0)
captured_faces = []
count = 0

while count < REQUIRED_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(rgb)
    except:
        faces = []

    for face in faces:
        if face["confidence"] >= FACE_CONFIDENCE:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            face_img = rgb[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img, (160, 160))
            captured_faces.append(face_img)
            count += 1

            print(f"✔ Captured {count}/{REQUIRED_IMAGES}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.waitKey(400)
            break

    cv2.putText(
        frame,
        f"Captured: {count}/{REQUIRED_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("❌ Registration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

print("\n🔄 Generating embeddings...")

# ================= GENERATE EMBEDDINGS =================
embeddings = []

for face in captured_faces:
    face = np.expand_dims(face, axis=0)
    emb = embedder.embeddings(face)[0]
    embeddings.append(emb)

embeddings = np.array(embeddings)
mean_embedding = np.mean(embeddings, axis=0)

# ================= SAVE EMBEDDING =================
np.save(embedding_path, mean_embedding)

print("✅ Registration successful!")
print(f"User '{user_id}' saved with FaceNet embedding.")
