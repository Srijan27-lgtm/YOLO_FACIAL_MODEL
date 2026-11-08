import cv2
import numpy as np
import os
import time
import pyttsx3
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download

print("🚀 Starting Ultra-Fast Voice Attendance System...")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_IMAGE_PATH = None  # Set to "images.jpg" to test a photo, or None for webcam

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech (words per minute)
engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)

# Voice Cooldown variables (prevents spamming)
last_spoken_time = 0
SPEAK_COOLDOWN = 3.0 # Seconds to wait before speaking again

print("⬇️ Checking/Downloading best.onnx from Hugging Face...")
# Download the optimized ONNX model
model_path = hf_hub_download(
    repo_id="srijan2715/yolo12m-face-detection", 
    filename="weights/best.onnx", # <-- CHANGED TO ONNX
    local_dir="." 
)
print("✅ Optimized ONNX Model ready!")

print("🧠 Loading AI Models...")
# Load the ONNX model (Ultralytics handles this seamlessly and runs it faster!)
yolo_model = YOLO(model_path, task='detect')

# Use CPU. (If you have an NVIDIA GPU, change to 'CUDAExecutionProvider')
arcface_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
arcface_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Models loaded.\n")

# ==========================================
# 🧠 CORE PROCESSING FUNCTION
# ==========================================
def process_frame(frame):
    global last_spoken_time
    
    embeddings_list = []
    
    # Run YOLO Detection (ONNX makes this faster)
    results = yolo_model.predict(source=frame, conf=0.5, imgsz=640, verbose=False)
    
    if len(results[0].boxes) > 0:
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            
            h, w = frame.shape[:2]
            pad = 0.30
            box_w = x2 - x1
            box_h = y2 - y1
            x1_pad = max(0, int(x1 - box_w * pad))
            y1_pad = max(0, int(y1 - box_h * pad))
            x2_pad = min(w, int(x2 + box_w * pad))
            y2_pad = min(h, int(y2 + box_h * pad))
            
            cv2.rectangle(frame, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 0), 2)
            cv2.putText(frame, f"Face #{idx+1} {conf:.2f}", (x1_pad, y1_pad - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cropped_face = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if cropped_face.shape[0] > 80 and cropped_face.shape[1] > 80:
                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                faces = arcface_app.get(cropped_face_rgb)
                
                if len(faces) > 0:
                    embedding = faces[0].embedding
                    embeddings_list.append({'face_id': idx + 1, 'embedding': embedding})
                    
                    cv2.rectangle(frame, (x1_pad, y1_pad), (x2_pad, y2_pad), (255, 0, 0), 3)
                    cv2.putText(frame, f"ID #{idx+1} Captured!", (x1_pad, y1_pad - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # === VOICE CONFIRMATION LOGIC ===
                    current_time = time.time()
                    if current_time - last_spoken_time > SPEAK_COOLDOWN:
                        # In the future, this will say the person's name from the database!
                        # For now, it confirms successful recognition.
                        text_to_speak = f"Identity {idx + 1} captured successfully."
                        print(f"🗣️ Speaking: {text_to_speak}")
                        engine.say(text_to_speak)
                        engine.runAndWait() # This blocks briefly to speak
                        last_spoken_time = current_time
                        
    return frame, embeddings_list

# ==========================================
# 🎬 MAIN EXECUTION LOOP
# ==========================================

if INPUT_IMAGE_PATH and os.path.exists(INPUT_IMAGE_PATH):
    print(f"🖼️ Loading image: {INPUT_IMAGE_PATH}")
    frame = cv2.imread(INPUT_IMAGE_PATH)
    if frame is not None:
        processed_frame, embeddings_list = process_frame(frame)
        cv2.imshow("Face Recognition Result", processed_frame)
        print("👉 Press ANY KEY to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("🎥 Starting Live Webcam (Edge AI Optimized + Voice)...")
    print("👉 Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, embeddings_list = process_frame(frame)
        
        if len(embeddings_list) > 0:
            cv2.putText(processed_frame, f"Faces: {len(embeddings_list)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Ultra-Fast Voice Attendance", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 System closed.")