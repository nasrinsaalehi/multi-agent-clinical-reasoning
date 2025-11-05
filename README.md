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

