from src.logger.basic_logging import logging
from src.data.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
import pandas as pd

from src.models.model_trainer import ModelTrainer
from src.pipelines.training_pipeline import CustomerChurnPipeline

try:
    CustomerChurnPipeline().run_pipeline()
except Exception as e:
    logging.error("Error here. Exiting", exc_info=e)