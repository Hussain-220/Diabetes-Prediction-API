# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load the local dataset
print("Loading local diabetes dataset...")
# Define column names for the Pima Indians Diabetes dataset
column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']

# Load the CSV file without headers and assign column names
df = pd.read_csv('diabetes.csv', header=None, names=column_names)
print(f"Dataset shape: {df.shape}")

# Split features and target
X = df.drop('outcome', axis=1)
y = df['outcome']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = rf_model.score(X_train_scaled, y_train)
test_score = rf_model.score(X_test_scaled, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Save the model and scaler
print("Saving model and scaler...")
model_data = {
    'model': rf_model,
    'scaler': scaler,
    'features': list(X.columns)
}

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model and scaler saved successfully as 'diabetes_model.pkl'")