from locust import HttpUser, task, between
from random import randint

class PredictEndpointUser(HttpUser):
    # Wait time between tasks (to simulate real user behavior)
    wait_time = between(0, 2)

    @task
    def predict(self):
        # Request payload for the `/predict` endpoint
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

        # Send a POST request to the `/predict` endpoint
        with self.client.post("/fastapp/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")

class SquareEndpointUser(HttpUser):
    # Wait time between tasks (to simulate real user behavior)
    wait_time = between(0, 0.1)

    @task
    def square(self):

        num: int = randint(1, 100)
        with self.client.get(f"/fastapp/square?number={num}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")