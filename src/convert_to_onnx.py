from ultralytics import YOLO
import os

print("🚀 Converting local best.pt to best.onnx for Edge AI Speed...")

# Find the downloaded .pt file
pt_path = "weights/best.pt"
if not os.path.exists(pt_path):
    pt_path = "best.pt" # Fallback if it's in the root folder

if os.path.exists(pt_path):
    print(f"Found model at: {pt_path}")
    model = YOLO(pt_path)
    
    # Export to ONNX format (this optimizes it for fast CPU/GPU inference)
    print("⏳ Converting... (this takes about 10-20 seconds)")
    model.export(format="onnx", imgsz=640, dynamic=True)
    
    print("\n🎉 SUCCESS! 'best.onnx' has been created in your folder.")
    print("You can now delete best.pt if you want to save space.")
else:
    print("❌ Could not find best.pt. Please run the webcam script once to download it first.")