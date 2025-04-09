# app.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
print("Loading model...")
with open('diabetes_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

@app.route('/')
def home():
    return "Diabetes Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        json_data = request.get_json()
        
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([json_data])
        
        # Make sure we have all required features in the right order
        input_df = pd.DataFrame(columns=features)
        for col in features:
            if col in input_data.columns:
                input_df[col] = input_data[col]
            else:
                return jsonify({
                    'status': 'error',
                    'message': f"Missing feature: {col}"
                })
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Probability of class 1
        
        # Return the prediction as JSON
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'probability': float(probability),
            'message': f"The patient {'has' if prediction == 1 else 'does not have'} diabetes"
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)