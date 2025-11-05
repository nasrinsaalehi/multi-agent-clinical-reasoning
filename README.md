# 🧠 Living Intelligence for Clinical Reasoning

## Overview
This project demonstrates the design and implementation of a **Living Intelligence system** — a multi-agent computational framework that performs **clinical reasoning**, **prediction**, and **explainability** for hospital decision-support.  
The system uses the **MIMIC-IV Clinical Database (demo version)** to model post-surgical outcomes such as **Length of Stay (LOS)** and integrates explainable AI, graph-based inference, and uncertainty estimation.

---

## 🏗️ System Architecture

| Agent | Function | Technologies |
|--------|-----------|--------------|
| **Perception Agent** | Extracts and normalizes structured + unstructured patient data | `pandas`, `gzip`, `tqdm` |
| **Inference Agent** | Predicts post-surgical complication risk or LOS | `scikit-learn`, `PyTorch`, `PyG` |
| **Explainability Agent** | Translates predictions into human-readable reasoning | `Python`, custom `stringify_*` functions |

The agents communicate via shared data artifacts (`.csv`, `.npy`, `.pkl` files) and maintain transparency through feature-level reasoning and uncertainty awareness.

---

## 📦 Repository Structure

```
├── 01-load_data.ipynb
├── 02-explainability.ipynb
├── 03-preprocess-dataset.ipynb
├── 04-prediction-length-of-stay.ipynb
├── 05-graph-based-prediction.ipynb
├── utils.py
├── data/
├── outputs/
├── Living_Intelligence_Results_and_Findings.docx
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/living-intelligence-clinical.git
cd living-intelligence-clinical
```

### 2. Create and Activate Conda Environment
```bash
conda create -n clinical-agent python=3.10
conda activate clinical-agent
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
torch
torch-geometric
scikit-learn
tqdm
jsonlines
dill
```

---

## 🚀 Run Flow

You can execute the workflow either manually step-by-step (Jupyter) or automatically using the provided shell script.

### **Automatic Run Flow**
```bash
bash run_flow.sh
```

### **Manual Execution**
```bash
# Step 1: Load and Normalize Data
jupyter nbconvert --to notebook --execute 01-load_data.ipynb

# Step 2: Preprocess Dataset
jupyter nbconvert --to notebook --execute 03-preprocess-dataset.ipynb

# Step 3: Predict Length of Stay
jupyter nbconvert --to notebook --execute 04-prediction-length-of-stay.ipynb

# Step 4: Graph-Based Prediction
jupyter nbconvert --to notebook --execute 05-graph-based-prediction.ipynb

# Step 5: Explainability Output
jupyter nbconvert --to notebook --execute 02-explainability.ipynb
```

---

## 🔍 Example Output
```
Patient 10004235 was seen at 08/09/2023 and given admission id 24181354.
The admission type was urgent. The means of arrival was transfer from hospital.
The patient's primary language was English. The patient's insurance was Medicaid.
Predicted LOS: 7.2 days ± 1.3 (uncertainty).
```

---

## 🧩 Ethical and Safety Considerations
- **Uncertainty-Aware Predictions:** The GNN model reports variance alongside mean prediction.  
- **Human Oversight:** Outputs are designed for *decision support*, not automation.  
- **Data Privacy:** Follows HIPAA and local data governance standards.  
- **Feedback Loop:** Clinician feedback is logged for retraining and self-correction.

---

## 🧠 Future Extensions
- Reinforcement learning for adaptive clinician feedback  
- Integration with FHIR-based hospital dashboards  
- Real-time monitoring for model drift and bias  

---

## 👩‍⚕️ Authors
**Nasrin Salehi**  
Research in clinical AI systems, agentic reasoning, and explainable health intelligence.

---

## 📜 License
Distributed under the MIT License.
