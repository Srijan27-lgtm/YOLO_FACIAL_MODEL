#  AI-Powered Real-Time Face Recognition Attendance System

An end-to-end, biometric attendance system that combines custom-trained object detection (YOLOv12m) with deep metric learning (ArcFace) to detect, align, and extract 512-dimensional facial embeddings in real-time.



Project Documentation

### 1. Abstract
Traditional attendance systems are prone to human error and proxy attendance. This project presents a robust, automated Face Recognition Attendance System. It leverages a two-stage computer vision pipeline: a custom-trained YOLOv12m model for real-time face detection, and the ArcFace neural network for generating unique 512-dimensional facial embeddings. The YOLOv12m model was trained on a curated subset of the WIDER FACE dataset, achieving a Mean Average Precision (mAP50) of 95%.

### 2. Approach & Methodology
To build a reliable system, we adopted a four-phase approach:
*   **Phase 1: Data Engineering:** Utilized the WIDER FACE dataset. Developed a custom Python script to filter out blurry/tiny faces, resulting in 12,880 high-quality images with 83,882 face annotations.
*   **Phase 2: Cloud Training:** Custom-trained YOLOv12m on Kaggle Cloud GPUs (Tesla T4) for 50 epochs.
*   **Phase 3: The Recognition Pipeline:** Integrated ArcFace (`buffalo_l`). Solved a critical facial alignment bottleneck by implementing a **dynamic 30% bounding-box padding algorithm**, ensuring ArcFace has enough peripheral context to detect landmarks on tightly cropped faces.
*   **Phase 4: Edge Deployment:** Exported PyTorch weights to ONNX format for hardware-agnostic, accelerated inference on local machines.

### 3.Architecture
The backend is split into three layers: a presentation layer (user interface), an application layer (Python modules) and a data layer (MySQL).

Camera → YOLO Detection → Preprocessing → Embedding → Matching → Attendance → MySQL

Module	Role
Camera	Reads frames from the local camera
Face Detection	YOLO locates faces and returns bounding boxes
Preprocessing	Crops and prepares each face for the model
Embedding Generation	Converts a face into a numerical vector
Face Matching	Compares with stored embeddings using cosine similarity
Attendance	Validates session, class and duplicates, then saves
Database	Runs all MySQL queries
### 4.Student Registration
Capture clear face images of the student.
Detect, crop and preprocess the face.
Generate the embedding and store it in face_embeddings against the student ID.
### 5.Live Attendance
Load all active embeddings (is_active = 1) from MySQL.
Capture a frame and detect faces with YOLO.
Generate an embedding for each face and compute cosine similarity with every stored embedding.
Accept the best match only if its score is above the threshold, otherwise mark the face as UNKNOWN.
Validate the session, check the student belongs to the class, and check attendance is not already marked.
Insert the attendance record inside a transaction.
### 6. Results
*   **Detection Accuracy:** 95% mAP50 on the custom WIDER FACE subset.
*   **Robustness:** The dynamic padding logic successfully resolved alignment errors, allowing reliable embedding extraction from angled or partially framed faces.
*   **Performance:** Real-time multi-face detection and embedding extraction.
Tables: students, teachers, departments, classes, sessions, class_session_join, face_embeddings, attendance. A unique constraint on (class_session_id, student_id) prevents duplicate rows.
