# 🧪 AI Drug Toxicity Predictor (Track A)

## 📌 Project Overview
Drug development frequently fails due to unexpected toxicity, costing billions and risking patient safety. This project is a machine learning prototype designed to predict potential drug toxicity early in the pipeline using chemical structures (SMILES strings) and molecular descriptor data. 

This repository contains a full end-to-end pipeline: from data preprocessing and RDKit feature engineering, to an XGBoost classification model, and finally an interactive Streamlit web interface for real-time predictions.

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.x
* **Cheminformatics:** RDKit (Molecular Fingerprint generation)
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Frontend/Prototype:** Streamlit

## ✨ Features
* **Automated Feature Engineering:** Converts raw SMILES chemical text into 1024-bit Morgan Fingerprints.
* **Scalable ML Model:** Utilizes XGBoost for high-accuracy, highly scalable tabular data classification.
* **Interactive Web Application:** A user-friendly Streamlit prototype allowing researchers to input a SMILES string and instantly receive a toxicity risk assessment and confidence score.
* **Best Practices:** Includes virtual environment support, clean git hygiene, and modularized code.

## ⚙️ Technical Workflow
1. **Data Ingestion:** Loads the Tox21 dataset containing ~8,000 compounds and 12 toxicity assay results.
2. **Preprocessing (`process_features.py`):** Consolidates 12 assays into a binary `is_toxic` target. Uses RDKit to map SMILES strings to mathematical Morgan Fingerprints.
3. **Model Training (`train_model.py`):** Splits data (80/20), trains an XGBoost classifier, and exports the serialized model (`.pkl`) for production.
4. **Deployment (`app.py`):** Streamlit loads the serialized model and processes real-time user input for instant inference.

## 🚀 Installation & Setup Instructions

To run this prototype locally on your machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/0908vishwanandan/ai-toxicity-predictor.git](https://github.com/0908vishwanandan/ai-toxicity-predictor.git)
cd ai-toxicity-predictor
