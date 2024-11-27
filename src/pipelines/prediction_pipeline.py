
import os
from src.logger.basic_logging import logging
import numpy as np
import pandas as pd
from src.utils.basic_util import load_object
from src.features.data_transformation import DataTransformation

class CustomerChurnPredictionPipeline:
    def __init__(self):
        self.transformer = DataTransformation()
        self.load_artifacts()
        logging.info("Model and Preprocessor loaded")

    # can allow reloading models
    def load_artifacts(self):
        preprocessor_path = os.path.join("data/processed", "preprocessor.pkl")
        model_path = os.path.join("models/trained", "model.pkl")
        self.processor = load_object(preprocessor_path)
        self.model = load_object(model_path)
        logging.info("Model and processor reloaded")

    def predict(self, features):

        scaled = self.transformer.apply_transforms(features, self.processor)
        logging.info(scaled)
        pred = self.model.predict(scaled)

        return pred

class CustomerData:
    def __init__(self, 
                customer_id: str,
                gender: str,
                senior_citizen: np.int64,
                partner: str,
                dependents: str,
                tenure: np.int64,
                phone_service: str,
                multiple_lines: str,
                internet_service: str,
                online_security: str,
                online_backup: str,
                device_protection: str,
                tech_support: str,
                streaming_tv: str,
                streaming_movies: str,
                contract: str,
                paperless_billing: str,
                payment_method: str,
                monthly_charges: np.float64,
                total_charges: str):
        
        self.customer_id = customer_id
        self.gender = gender
        self.senior_citizen = senior_citizen
        self.partner = partner
        self.dependents = dependents
        self.tenure = tenure
        self.phone_service = phone_service
        self.multiple_lines = multiple_lines
        self.internet_service = internet_service
        self.online_security = online_security
        self.online_backup = online_backup
        self.device_protection = device_protection
        self.tech_support = tech_support
        self.streaming_tv = streaming_tv
        self.streaming_movies = streaming_movies
        self.contract = contract
        self.paperless_billing = paperless_billing
        self.payment_method = payment_method
        self.monthly_charges = monthly_charges
        self.total_charges = total_charges



    def get_pd_dataframe(self):
        input = {
            'customerID': [self.customer_id],
            'gender': [self.gender],
            'SeniorCitizen': [self.senior_citizen],
            'Partner': [self.partner],
            'Dependents': [self.dependents],
            'tenure': [self.tenure],
            'PhoneService': [self.phone_service],
            'MultipleLines': [self.multiple_lines],
            'InternetService': [self.internet_service],
            'OnlineSecurity': [self.online_security],
            'OnlineBackup': [self.online_backup],
            'DeviceProtection': [self.device_protection],
            'TechSupport': [self.tech_support],
            'StreamingTV': [self.streaming_tv],
            'StreamingMovies': [self.streaming_movies],
            'Contract': [self.contract],
            'PaperlessBilling': [self.paperless_billing],
            'PaymentMethod': [self.payment_method],
            'MonthlyCharges': [self.monthly_charges],
            'TotalCharges': [self.total_charges]
        }

        data= pd.DataFrame(input)
        logging.info(f"Data: {data.to_dict()}")
        return data

