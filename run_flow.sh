#!/bin/bash
# ======================================================
# Living Intelligence Project: Automated Run Flow
# ======================================================
echo "Initializing Living Intelligence Workflow..."

echo "Step 1: Loading and Normalizing Data"
jupyter nbconvert --to notebook --execute 01-load_data.ipynb

echo "Step 2: Preprocessing Dataset"
jupyter nbconvert --to notebook --execute 03-preprocess-dataset.ipynb

echo "Step 3: Predicting Length of Stay (LOS)"
jupyter nbconvert --to notebook --execute 04-prediction-length-of-stay.ipynb

echo "Step 4: Running Graph-Based Prediction Model"
jupyter nbconvert --to notebook --execute 05-graph-based-prediction.ipynb

echo "Step 5: Generating Explainability Outputs"
jupyter nbconvert --to notebook --execute 02-explainability.ipynb

echo "Workflow completed successfully!"
