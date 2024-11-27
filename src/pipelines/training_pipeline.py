from dataclasses import dataclass
from src.data.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
from src.models.model_trainer import ModelTrainer, ModelTrainingConfig
from src.logger.basic_logging import logging
import mlflow

@dataclass
class MlFlowConfig:
    mlflow_uri = "http://localhost:5000"
    experiment_name = "MLOPS-Workflow-5"


class CustomerChurnPipeline:
    def __init__(self) -> None:
        mlflow_config = MlFlowConfig()
        mlflow.set_tracking_uri(mlflow_config.mlflow_uri)
        mlflow.set_experiment(mlflow_config.experiment_name)


    def run_pipeline(self):
        try:

            # Start MLflow run for entire pipeline
            with mlflow.start_run(run_name="complete_pipeline"):
                # Data Ingestion
                logging.info("Starting Data Ingestion")
                data_ingestion = DataIngestion()   
                train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
                mlflow.log_param("train_data_path", train_data_path)
                mlflow.log_param("test_data_path", test_data_path)

                # Data Transformation
                logging.info("Starting Data Transformation")
                data_transformation = DataTransformation()
                X_train, y_train, X_test, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
                    train_data_path, test_data_path
                )
                mlflow.log_param("preprocessor_path", preprocessor_path)

                # Model Training
                logging.info("Starting Model Training")
                model_trainer = ModelTrainer()
                accuracy = model_trainer._train_models(X_train, y_train, X_test, y_test)
                
                logging.info(f"Training pipeline completed. Best model accuracy: {accuracy}")
                return accuracy
            
        except Exception as e:
            logging.error("Error occurred during training", exc_info=e)
            exit(1)