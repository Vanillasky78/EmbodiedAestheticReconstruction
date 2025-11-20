# Embodied Aesthetic Reconstruction  
*Final MSc Project — University of the Arts London (CCI)*  

## Project Statement  
"Embodied Aesthetic Reconstruction Against Disciplinary Norms" is a research project on a personalised artistic image-generation system based on motion capture and generative AI, creating playful and artistic outputs that correspond to each individual’s body shape. 
It aims to break the body anxiety caused by standardised and homogenised ideals of the “perfect body.” By using motion-capture technology to scan each person’s physical form, and an AI-driven image database to automatically match suitable artworks (or fashion silhouettes), the project highlights that every individual possesses their own complete aesthetic presence and value. What is considered perfect (a beautiful body) should inherently be diverse and unique.  

## 1. System Overview  
This MVP implements a fully functional real-time pose-driven curatorial AI system combining:

• YOLOv8-Pose for live keypoint detection  
• OpenCLIP (ViT-B/32) for image embeddings  
• Hybrid similarity scoring  
• Curated multi-museum image dataset  
• Streamlit frontend + FastAPI backend  
• Stillness-triggered auto-capture (3–5 seconds)  
• Real-time artwork display with metadata (artist / year / value)  

Pipeline:  
Camera → Pose Estimation → Embedding Fusion → Artwork Matching → Curated Output

## 2. Repository Structure
EmbodiedAestheticReconstruction/
│
├── backend/                     # FastAPI backend engine
│   ├── main.py                  # API entrypoint
│   ├── config.py
│   ├── utils_pose.py            # Keypoint → vector encoder
│   ├── model/
│   │   ├── pose_matcher.py      # Hybrid pose + CLIP matcher
│   │   ├── utils.py
│   │   └── yolov8n-pose.pt
│   └── tools/
│       ├── build_embeddings.py
│       ├── build_pose_embeddings.py
│       └── build_mixed_index.py
│
├── frontend/                    # Streamlit UI
│   ├── app_frontend.py
│   ├── app_curatorial.py
│   └── yolov8n-pose.pt
│
├── data/                        # Art datasets (local + MET + AIC)
│   ├── mixed/
│   │   ├── images/
│   │   ├── embeddings.npy
│   │   ├── pose_embeddings.npy
│   │   └── embeddings_meta.csv
│   ├── local/
│   ├── met/
│   └── aic/
│
├── scripts/
│   ├── setup_check.py
│   ├── start_local.sh
│   └── start_local.bat
│
├── README.md
└── requirements.txt

## 3. Installation
1. Clone the repository

git clone https://github.com/Vanillasky78/EmbodiedAestheticReconstruction.git
cd EmbodiedAestheticReconstruction

2. Create environment (Python 3.10 recommended)

conda create -n ear-mvp python=3.10 -y
conda activate ear-mvp

3. Install backend + frontend dependencies

pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

## 4. Running the System
1. Start backend:

uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
2. Start Frontend:

streamlit run frontend/app_frontend.py
3. The app will open at:

http://localhost:8501

## 5. How It Works
1. Motion Capture  
   YOLOv8-Pose extracts 17 keypoints from camera input in real time.

2. Stillness Detection  
   A 3–5 second stability window auto-triggers the capture.

3. Embedding Fusion  
   • CLIP embedding → semantic meaning  
   • Pose embedding → structural geometry  

4. Database Matching  
   Each artwork includes:  
   - CLIP embedding  
   - Pose embedding  
   - Metadata (artist, title, year)  
   - Optional price estimation  

5. Hybrid Similarity  

final_score = 0.65 * CLIP + 0.35 * Pose

6. Output  
   The matched artwork appears on the right panel with:  
   - Yellow label (value, year, artist)  
   - Pose overlay  
   - Symmetric full-screen layout

## 6. Optional: Rebuild Indexes
1. Build CLIP embeddings
python -m backend.tools.build_embeddings --museum_dir data/local

2. Build pose embeddings
python -m backend.tools.build_pose_embeddings --museum_dir data/local --device cpu

3. Build mixed index
python -m backend.tools.build_mixed_index

## 7. Data Requirements
Each dataset folder (local / met / aic) must contain:

data/<museum>/
│   images/
│   embeddings.npy
│   pose_embeddings.npy
│   embeddings_meta.csv

The system defaults to:
data/mixed/

## 8. System Architecture (Text Diagram)
┌────────────────────────────┐
│        👩 Audience          │
│  Moves or stands in front  │
│  of camera (pose changes)  │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│   🎨 Streamlit Frontend (UI)
│  1️⃣ Captures camera input
│  2️⃣ Detects stillness
│  3️⃣ Sends image to API
│  4️⃣ Displays matched artwork
│
│  Modes:
│   • Local (PoseMatcher)
│   • Remote (FastAPI)
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│   ⚙️ FastAPI Backend
│  Receives image → matching:
│   1️⃣ YOLOv8-Pose
│   2️⃣ OpenCLIP
│   3️⃣ Hybrid similarity
│
│  Endpoints:
│   - /match
│   - /list_museums
│   - /metadata/{museum}/{file}
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│   🗂️ Data Layer
│  - images/
│  - embeddings.npy
│  - pose_embeddings.npy
│  - embeddings_meta.csv
└────────────────────────────┘

## 9. Outcome Demonstration
https://vimeo.com/1138944508

## 10. Features
- **Motion Capture Input** → extract pose landmarks (via MediaPipe).
- **Cross-modal Embeddings** → fuse pose with personality traits (Big Five sliders).
- **Generative Output**:
  - *Visual*: Stylised image output (Diffusion/Dreambooth optional).
  - *Audio*: Soundscape generation (RAVE/AudioLM integration planned).
- **Ethics by Design**: consent gate, data retention toggle, bias awareness.

## 11. Credit
Author: Xinyi Zhang  
Programme: MSc Computing and Creative Industry  
Institute: UAL – Creative Computing Institute  
Year: 2025  
