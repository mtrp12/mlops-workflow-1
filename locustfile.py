from locust import HttpUser, task, between
import json

class PredictEndpointUser(HttpUser):
    # Wait time between tasks (to simulate real user behavior)
    wait_time = between(0, 0.1)

    @task
    def predict(self):
        # Request payload for the `/predict` endpoint
        payload = {
            "feature1": 1.2,
            "feature2": 3.4,
            "feature3": 5.6
        }

        # Send a POST request to the `/predict` endpoint
        with self.client.get("/predict", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")
