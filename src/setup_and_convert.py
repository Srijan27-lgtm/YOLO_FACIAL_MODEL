from ultralytics import YOLO
import os
from huggingface_hub import hf_hub_download

# 1. EXPLICITLY DEFINE YOUR EXACT FOLDER PATH
TARGET_DIR = r"C:\Users\b9499\OneDrive\Desktop\FaceAttendance"

print(f"🚀 Setting up Edge AI Optimized Model...")
print(f"📂 Target Directory: {TARGET_DIR}\n")

# Ensure the directory exists
os.makedirs(TARGET_DIR, exist_ok=True)

# 2. Find or Download best.pt
# Check both the root folder and the weights subfolder
pt_path_root = os.path.join(TARGET_DIR, "best.pt")
pt_path_weights = os.path.join(TARGET_DIR, "weights", "best.pt")

if os.path.exists(pt_path_weights):
    pt_path = pt_path_weights
elif os.path.exists(pt_path_root):
    pt_path = pt_path_root
else:
    print("⬇️ 'best.pt' not found locally. Downloading from Hugging Face...")
    try:
        # Download directly into your target directory
        downloaded_path = hf_hub_download(
            repo_id="srijan2715/yolo12m-face-detection", 
            filename="weights/best.pt",
            local_dir=TARGET_DIR 
        )
        pt_path = downloaded_path
        print(f"✅ Download complete! Saved to: {pt_path}")
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        print("Please check your internet connection.")
        exit()

print(f"📂 Using model at: {pt_path}")

# 3. Convert to ONNX explicitly in your folder
onnx_path = os.path.join(TARGET_DIR, "best.onnx")
print(f"⏳ Converting to ONNX format for Edge AI Speed (this takes ~15-30 seconds)...")

# Load the model
model = YOLO(pt_path)

# Export to ONNX. Ultralytics will save it right next to the .pt file.
exported_path = model.export(format="onnx", imgsz=640, dynamic=True)

# 4. Verify and Report
if os.path.exists(exported_path):
    print(f"\n🎉 SUCCESS! Model exported successfully.")
    print(f"💾 Your ultra-fast model is located at: {exported_path}")
    print("💡 You can now run your main webcam_test.py script!")
else:
    print(f"\n⚠️ Conversion finished, but the file was not found at: {onnx_path}")
    print("Please check the folder manually.")