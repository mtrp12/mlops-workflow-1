import os
from sklearn.model_selection import train_test_split
import pandas as pd
from src.logger.basic_logging import logging
from src.data.download_dataset import download_dataset
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("data/raw", "train.csv")
    test_data_path = os.path.join("data/raw", "test.csv")
    raw_data_path = os.path.join("data/raw", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingesion has been started")
        try: 
            logging.info("Downloading data from kaggle")
            download_dataset()
            
            logging.info("Reading downloaded raw data into pandas dataframe")
            data = pd.read_csv(os.path.join("data/raw", "raw.csv"))
            logging.info("Raw data reading has been completed")

            train_set, test_set = train_test_split(data, test_size= .20, random_state=2)
            logging.info("Raw data has been splitted into Train and Test set")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion has been completed!")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path 

      
        except Exception as e:
            logging.error("Error occurred while ingesting data", exc_info=e)

if __name__ == "__main__":
    pass