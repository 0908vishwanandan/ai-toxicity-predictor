import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle

# 1. Load the processed data from our previous step
print("Loading processed data...")
df = pd.read_pickle('processed_data.pkl')

# 2. Prepare X (Features/Fingerprints) and y (Target/Toxicity)
print("Structuring data for machine learning...")
X = np.vstack(df['fingerprint'].values)
y = df['is_toxic'].values

# 3. Split into training (80%) and testing (20%) sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the XGBoost Model
print("Training XGBoost Model (this may take 30-60 seconds)...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model's functionality
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model so our web app prototype can use it later
print("\nSaving model for the web application prototype...")
with open('toxicity_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Success! Model saved as 'toxicity_model.pkl'")