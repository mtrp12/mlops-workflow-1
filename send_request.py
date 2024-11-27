import requests
import json


payload = {
    "customerID": "9305-CDSKC",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 8,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 99.65,
    "TotalCharges": "820.5"
}

# Send POST request
response = requests.post(
    "http://localhost:8000/predict",
    json=payload,
    headers={"Content-Type": "application/json"}
)

# Print results
print("Status Code:", response.status_code)
print("Response:", response.json())