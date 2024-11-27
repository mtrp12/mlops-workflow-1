import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
from src.logger.basic_logging import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.basic_util import save_object
from xgboost import XGBClassifier

@dataclass
class ModelTrainingConfig:
    train_model_file_path = os.path.join("models/trained", "model.pkl")
    # mlflow_uri = "http://localhost:5000"
    # experiment_name = "MLOPS-Workflow-4"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

        # # Configure MLFlow
        # mlflow.set_tracking_uri(self.model_trainer_config.mlflow_uri)
        # mlflow.set_experiment(self.model_trainer_config.experiment_name)

        # self.client = MlflowClient()
        self.run_name = f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _log_metrics(self, y_true, y_pred, prefix=""):
        """Calculate and log metric to MLFlow"""
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred, average='weighted'),
            f"{prefix}recall": recall_score(y_true, y_pred, average='weighted'),
            f"{prefix}f1": f1_score(y_true, y_pred, average='weighted'),
        }
        mlflow.log_metrics(metrics)
        return metrics
    
    def _train_model(self, X_train, y_train, X_test, y_test, model_name, model, params):
        """Train and evaluate a single model with grid search"""
        try:
            with mlflow.start_run(run_name=f"{model_name}_{self.run_name}", nested=True) as run:
                logging.info(f"Started Training {model_name}")

                # Log model parameters
                mlflow.log_params(params)

                # Perform Grid Search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=10,
                    n_jobs=1,
                    verbose=2,
                    scoring='accuracy'
                )

                # Train the Model
                grid_search.fit(X_train, y_train)

                # Log best parameters
                best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
                mlflow.log_params(best_params)
                logging.info(f"Best Paramaters for {model_name}: {best_params}")

                # Get Predictions
                y_train_pred = grid_search.predict(X_train)
                y_test_pred = grid_search.predict(X_test)
                
                # Log Metrics
                train_metrics = self._log_metrics(y_train, y_train_pred, prefix="train_")
                test_metrics = self._log_metrics(y_test, y_test_pred, prefix="test_")

                # Log CV Scores
                mlflow.log_metric("cv_mean_score", grid_search.best_score_)
                mlflow.log_metric("cv_std_score", grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                
                # Log feature importance (if available)
                # TODO: possible to get the actual names?
                if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': [f"feature_{i}" for i in range(X_train.shape[1])],
                        'importance': grid_search.best_estimator_.feature_importances_
                    })
                
                    # Create feature importance plot
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.bar(feature_importance['feature'], feature_importance['importance'])
                    plt.xticks(rotation=45)
                    plt.title(f'Feature Importance - {model_name}')
                    plt.tight_layout()
                    
                    # Save and log plot
                    plot_path = f"feature_importance_{model_name}.png"
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path)
                    os.remove(plot_path)
                
                # Model logging
                mlflow.sklearn.log_model(
                    grid_search.best_estimator_,
                    f"{model_name}_model",
                    registered_model_name=model_name
                )

                logging.info(f"Completed Training {model_name}")
                return grid_search.best_estimator_, test_metrics['test_accuracy']
        
        except Exception as e:
            logging.error("Error occurred during training {model_name}", exc_info=e)
            exit(1)
    
    def _train_models(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training pipeline")

            # Define models and parameters
            models = {
                "XGBClassifier": {
                    "model": XGBClassifier(eval_metric='logloss'),
                    "params": {
                        'n_estimators': [100, 500, 1000],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                },
                "RandomForest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "class_weight": ["balanced"],
                        'n_estimators': [20, 50, 30],
                        'max_depth': [10, 8, 5],
                        'min_samples_split': [2, 5, 10],
                    }
                },
                "DecisionTree": {
                    "model": DecisionTreeClassifier(),
                    "params": {
                        "class_weight": ["balanced"],
                        "criterion": ['gini', "entropy", "log_loss"],
                        "max_depth": [3, 4, 5, 6],
                        "min_samples_split": [2, 3, 4, 5],
                    }
                },
                "LogisticRegression": {
                    "model": LogisticRegression(),
                    "params": {
                        "class_weight": ["balanced"],
                        'C': [0.001, 0.01, 0.1, 1, 10],
                        'solver': ['liblinear', 'saga']
                    }
                }
            }
            
            # Training the models and store results
            model_results = {}
            for model_name, config in models.items():
                logging.info(f"Training {model_name}")
                model, accuracy = self._train_model(
                    X_train, y_train,
                    X_test, y_test,
                    model_name,
                    config['model'],
                    config['params']
                )
                model_results[model_name] = {
                    "model": model,
                    "accuracy":accuracy
                }
            
            best_model_name = max(model_results.items(), key=lambda x: x[1]["accuracy"])[0]
            best_model = model_results[best_model_name]["model"]
            best_accuracy = model_results[best_model_name]["accuracy"]

            logging.info(f"Best model: {best_model_name} with accuracy:{best_accuracy}")

            # Log best model summary
            with mlflow.start_run(run_name=f"best_model_summary_{self.run_name}", nested=True):
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_accuracy", best_accuracy)
                
                # Log comparison metrics
                comparison_metrics = {
                    f"{name}_accuracy": results["accuracy"]
                    for name, results in model_results.items()
                }
                mlflow.log_metrics(comparison_metrics)

                # Create comparison plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.bar(comparison_metrics.keys(), comparison_metrics.values())
                plt.xticks(rotation=45)
                plt.title('Model Comparison')
                plt.tight_layout()
                plt.savefig("model_comparison.png")
                mlflow.log_artifact("model_comparison.png")
                os.remove("model_comparison.png")

            # Save best model
            os.makedirs(os.path.dirname(self.model_trainer_config.train_model_file_path), exist_ok=True)
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            return best_accuracy
        
        except Exception as e:
            logging.error("Error in training pipeline", exc_info=e)
            exit(1)
        
    def run_training_pipeline(self):
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

    if __name__ == "__main__":
        # First start MLFLow server
        # mlflow server --host 0.0.0.0 --port 5000
        pass