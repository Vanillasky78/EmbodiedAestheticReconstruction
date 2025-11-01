# Embodied Aesthetic Reconstruction  
*Final MSc Project — University of the Arts London (CCI)*  

## 🌐 Overview  
This project, **Embodied Aesthetic Reconstruction**, explores how **motion capture** and **generative AI** can be used to resist disciplinary norms of body aesthetics.  
Instead of enforcing an “ideal body” shaped by algorithms, the system generates **playful, personalised artistic representations** that celebrate diverse embodiment.  

## ✨ Features  
- **Motion Capture Input** → extract pose landmarks (via MediaPipe).  
- **Cross-modal Embeddings** → fuse pose with personality traits (Big Five sliders).  
- **Generative Output**:  
  - *Visual*: Stylised image output (Diffusion/Dreambooth can be integrated).  
  - *Audio*: Soundscape generation (RAVE/AudioLM integration planned).  
- **Ethics by Design**: consent gate, data retention toggle, bias awareness.  

## 🚀 Installation  
Clone the repository and set up environment:  
```bash
git clone https://github.com/Vanillasky78/EmbodiedAestheticReconstruction.git
cd EmbodiedAestheticReconstruction
conda create -n ear-mvp python=3.10 -y
conda activate ear-mvp
pip install -r requirements.txt

streamlit run app.py

embodied-aesthetic-reconstruction/
├── app.py                  # Streamlit entrypoint
├── core/                   # Pose, personalisation, style selection, image ops
├── audio/                  # Embedding → audio synthesis
├── ui/                     # Streamlit UI components
├── ethics/                 # Consent text & ethics logic
├── tests/                  # Unit tests
├── outputs/                # Generated media (gitignored)
├── requirements.txt        # Dependencies
└── README.md               # Project description


cd ~/Documents/GitHub/EmbodiedAestheticReconstruction
conda env create -f environment.yml
conda activate ear-mvp
streamlit run app.py
------
conda activate ear-mvp
cd ~/Documents/GitHub/EmbodiedAestheticReconstruction
streamlit run app.py
