import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore') # Hides annoying RDKit warnings

print("Loading data...")
df = pd.read_csv('tox21.csv')

# 1. Combine the 12 tests into one 'is_toxic' score
toxicity_columns = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
                    'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
                    'SR-HSE', 'SR-MMP', 'SR-p53']

# If any test is 1.0, mark as 1.0 (Toxic). Otherwise 0.0 (Safe).
df['is_toxic'] = df[toxicity_columns].max(axis=1)

# Drop any rows where we have absolutely no test data
df = df.dropna(subset=['is_toxic'])

print(f"Total valid molecules: {len(df)}")
print(f"Toxic molecules (1.0): {sum(df['is_toxic'] == 1.0)}")
print(f"Safe molecules (0.0): {sum(df['is_toxic'] == 0.0)}")

# 2. Convert text SMILES into Number Fingerprints
print("\nConverting chemical structures to AI fingerprints...")
print("(This might take 10-30 seconds, please wait!)")

def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Generate a 1024-bit fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return np.array(fp)
    except:
        return None

df['fingerprint'] = df['smiles'].apply(smiles_to_fingerprint)

# Drop any broken molecules
df = df.dropna(subset=['fingerprint'])

print(f"Successfully converted {len(df)} molecules!")

# 3. Save this clean, AI-ready data so we don't have to rebuild it every time
df.to_pickle('processed_data.pkl')
print("\nSuccess! Saved clean data to 'processed_data.pkl'")