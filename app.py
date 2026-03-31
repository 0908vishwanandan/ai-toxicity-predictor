import streamlit as st
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Page Configuration
st.set_page_config(page_title="Toxicity Predictor", page_icon="🧪", layout="centered")

# 2. Load the trained AI model
@st.cache_resource # This makes the app run much faster
def load_model():
    with open('toxicity_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# 3. Function to translate user text into AI fingerprints
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Generate the exact same 1024-bit fingerprint we used for training
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    # Reshape it so XGBoost knows it's a single prediction
    return np.array(fp).reshape(1, -1) 

# 4. Design the User Interface
st.title("🧪 AI Drug Toxicity Predictor")
st.markdown("""
**Hackathon Prototype - Track A** This tool uses a machine learning model (XGBoost) trained on the Tox21 dataset to predict the potential toxicity of chemical compounds based on their molecular structure.
""")

st.divider()

# Input box for the user
st.markdown("### Enter Chemical Structure")
st.markdown("Paste a SMILES string below. *(Example: `CCOc1ccc2nc(S(N)(=O)=O)sc2c1`)*")
smiles_input = st.text_input("SMILES String:", "")

# 5. Prediction Logic
if st.button("Predict Toxicity", type="primary"):
    if smiles_input.strip() == "":
        st.warning("Please enter a SMILES string first.")
    else:
        with st.spinner("Analyzing molecular structure..."):
            fp = get_fingerprint(smiles_input)
            
            if fp is None:
                st.error("❌ Invalid SMILES string. The RDKit engine could not read this molecule.")
            else:
                # Get prediction and confidence score
                prediction = model.predict(fp)[0]
                probabilities = model.predict_proba(fp)[0]
                
                st.divider()
                st.markdown("### Prediction Results")
                
                if prediction == 1.0:
                    confidence = probabilities[1] * 100
                    st.error(f"⚠️ **High Risk of Toxicity**")
                    st.write(f"The AI is **{confidence:.2f}%** confident that this compound exhibits toxic traits.")
                else:
                    confidence = probabilities[0] * 100
                    st.success(f"✅ **Safe / Low Risk**")
                    st.write(f"The AI is **{confidence:.2f}%** confident that this compound is safe.")