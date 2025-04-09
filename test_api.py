# test_api.py
import requests
import json

# Sample data for diabetes prediction
diabetes_data = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50
}

# Send a POST request to the API
response = requests.post('http://localhost:5000/predict', 
                        json=diabetes_data, 
                        headers={'Content-Type': 'application/json'})

# Print the response
print(response.json())