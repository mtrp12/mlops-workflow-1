from src.logger.basic_logging import logging
from src.data.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
import pandas as pd

try:
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    x_train, y_train, x_test, y_test, preprocess_path = data_transformation.initiate_data_transformation(train_path, test_path)
except Exception as e:
    logging.error("Error here. Exiting", exc_info=e)