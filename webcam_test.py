import cv2
import numpy as np
import os
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download

print("🚀 Starting Stable Face Recognition System...")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_IMAGE_PATH = None  # Set to "images.jpg" to test a photo, or None for webcam

print("⬇️ Checking/Downloading best.pt from Hugging Face...")
model_path = hf_hub_download(
    repo_id="srijan2715/yolo12m-face-detection", 
    filename="weights/best.pt",
    local_dir="." 
)
print("✅ Model ready!")

print("🧠 Loading AI Models...")
# Load the standard, highly stable .pt model
yolo_model = YOLO(model_path)

# Use CPU for ArcFace
arcface_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
arcface_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Models loaded. Ready!\n")

# ==========================================
# 🧠 CORE PROCESSING FUNCTION
# ==========================================
def process_frame(frame):
    """Runs YOLO detection and ArcFace embedding extraction on ALL faces in frame."""
    embeddings_list = []
    
    # Run YOLO Detection
    results = yolo_model.predict(source=frame, conf=0.5, imgsz=640, verbose=False)
    
    if len(results[0].boxes) > 0:
        # Loop through ALL detected faces
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            
            # Add 30% padding for ArcFace
            h, w = frame.shape[:2]
            pad = 0.30
            box_w = x2 - x1
            box_h = y2 - y1
            x1_pad = max(0, int(x1 - box_w * pad))
            y1_pad = max(0, int(y1 - box_h * pad))
            x2_pad = min(w, int(x2 + box_w * pad))
            y2_pad = min(h, int(y2 + box_h * pad))
            
            # Draw Green Box (YOLO Detection)
            cv2.rectangle(frame, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 0), 2)
            cv2.putText(frame, f"Face #{idx+1} {conf:.2f}", (x1_pad, y1_pad - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Crop and run ArcFace
            cropped_face = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if cropped_face.shape[0] > 80 and cropped_face.shape[1] > 80:
                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                faces = arcface_app.get(cropped_face_rgb)
                
                if len(faces) > 0:
                    embedding = faces[0].embedding
                    embeddings_list.append({
                        'face_id': idx + 1,
                        'embedding': embedding,
                        'bbox': (x1_pad, y1_pad, x2_pad, y2_pad)
                    })
                    
                    # Draw Blue Box (ArcFace Success)
                    cv2.rectangle(frame, (x1_pad, y1_pad), (x2_pad, y2_pad), (255, 0, 0), 3)
                    cv2.putText(frame, f"ID #{idx+1} Captured!", (x1_pad, y1_pad - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # On-screen stats
                    cv2.putText(frame, f"Dim: 512", (x1_pad, y2_pad + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, f"Mean: {np.mean(embedding):.3f}", (x1_pad, y2_pad + 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame, embeddings_list

def print_embedding_details(embedding, face_id):
    """Prints the 512-D embedding neatly to the console."""
    np.set_printoptions(threshold=1000, linewidth=75) # Keep it clean, no massive spam
    
    print("\n" + "="*70)
    print(f"🧬 FACE #{face_id} EMBEDDING SUCCESSFULLY EXTRACTED!")
    print("="*70)
    print(f"📏 Shape: {embedding.shape}")
    print(f"📊 Stats - Min: {np.min(embedding):.4f} | Max: {np.max(embedding):.4f} | Mean: {np.mean(embedding):.4f}")
    print("\n🔢 First 15 values:")
    print(np.array2string(embedding[:15], separator=', '))
    print("\n🔢 Last 15 values:")
    print(np.array2string(embedding[-15:], separator=', '))
    print("="*70 + "\n")

# ==========================================
# 🎬 MAIN EXECUTION LOOP
# ==========================================

# MODE 1: Single Image Processing
if INPUT_IMAGE_PATH and os.path.exists(INPUT_IMAGE_PATH):
    print(f"🖼️ Loading image: {INPUT_IMAGE_PATH}")
    frame = cv2.imread(INPUT_IMAGE_PATH)
    
    if frame is not None:
        processed_frame, embeddings_list = process_frame(frame)
        
        if len(embeddings_list) > 0:
            print(f"\n🎉 SUCCESS! Detected {len(embeddings_list)} face(s) in the image!\n")
            for face_data in embeddings_list:
                print_embedding_details(face_data['embedding'], face_data['face_id'])
            
            output_name = "processed_" + os.path.basename(INPUT_IMAGE_PATH)
            cv2.imwrite(output_name, processed_frame)
            print(f"💾 Visual result saved as: {output_name}")
        else:
            print("❌ No valid faces detected in this image.")
            
        cv2.imshow("Face Recognition Result", processed_frame)
        print("👉 Press ANY KEY in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Failed to load image. Check the file path.")

# MODE 2: Live Webcam Processing
else:
    print("🎥 Starting Live Webcam (Stable Mode)...")
    print("👉 Press 'q' to quit | Press 's' to save current embeddings to file\n")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not access webcam.")
            break

        processed_frame, embeddings_list = process_frame(frame)
        
        if len(embeddings_list) > 0:
            cv2.putText(processed_frame, f"Faces: {len(embeddings_list)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Stable Face Attendance System", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(embeddings_list) > 0:
            for face_data in embeddings_list:
                filename = f"face_{face_data['face_id']}_embedding.npy"
                np.save(filename, face_data['embedding'])
                print(f"💾 Face #{face_data['face_id']} embedding saved to '{filename}'")
                print_embedding_details(face_data['embedding'], face_data['face_id'])

    cap.release()
    cv2.destroyAllWindows()
    print("👋 System closed.")