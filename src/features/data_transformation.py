import os
import sys
import pandas as pd
import numpy as np
from src.logger.basic_logging import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.basic_util import save_object
from feast import Field, FeatureStore, Entity, FeatureView, FileSource
from feast.types import Int64, String, Float64, Int32
from feast.value_type import ValueType
from datetime import datetime, timedelta
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("data/processed", "preprocessor.pkl")
    feature_store_repo_path = "feature_repo"

class DataTransformation:
    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
            
            # Get absolute path and create directory structure
            repo_path = os.path.abspath(self.data_transformation_config.feature_store_repo_path)
            os.makedirs(os.path.join(repo_path, "data"), exist_ok=True)
            
            # Create feature store yaml with minimal configuration
            feature_store_yaml_path = os.path.join(repo_path, "feature_store.yaml")
            
            # Simplified, minimal feature store configuration
            feature_store_yaml = """\
project: churn_prediction
provider: local
registry: data/registry.db
online_store:
    type: sqlite
offline_store:
    type: file
entity_key_serialization_version: 2\
            """
            
            # Write configuration file
            with open(feature_store_yaml_path, 'w') as f:
                f.write(feature_store_yaml)
            
            logging.info(f"Created feature store configuration at {feature_store_yaml_path}")

            # Verify the configuration file content
            with open(feature_store_yaml_path, 'r') as f:
                logging.info(f"Configuration file content:\n{f.read()}")
            
            # Initialize feature store
            self.feature_store = FeatureStore(repo_path=repo_path)
            logging.info("Feature store initialized successfully")

        except Exception as e:
            logging.error("Error in initialization. Exiting program.", exc_info=e)
            exit(1)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            le, mms, over = self._get_data_transformation_objects()
            logging.info("Obtained preprocessing object")
            
            logging.info("==========================")
            logging.info("Transforming training data")
            logging.info("==========================")
            train_data, x_train, y_train, le = self._transform_data(train_data, le, mms, over, train=True)

            logging.info("==========================")
            logging.info("Transforming testing data")
            logging.info("==========================")
            test_data, x_test, y_test, le = self._transform_data(test_data, le, mms, over, False)


            logging.info("Starting feature store operations")
            
            # Push data to Feast feature store
            self._push_features_to_store(train_data, "train")
            logging.info("Pushed training data to feature store")
            
            self._push_features_to_store(test_data, "test")
            logging.info("Pushed testing data to feature store")

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj={"le": le, "mms": mms, "over": over}
            )
            logging.info("Saved preprocessor objects")

            logging.info("Data transformation complete")
            return x_train, y_train, x_test, y_test, self.data_transformation_config.preprocess_obj_file_path

        except Exception as e:
            logging.error("Error in data transformation", exc_info=e)
            exit(1)

    def _get_data_transformation_objects(self):
        le = LabelEncoder()
        mms = MinMaxScaler()
        over = SMOTE(sampling_strategy=1)
        return le,mms,over

    def _transform_data(self, data, le, mms, over, train=False):
        data = self._cast_totalcharges_to_float(data)
        logging.info("Casted TotalCharges to float")

        data.drop(columns = ['customerID'], inplace = True)
        logging.info("Dropped customerID")

        data, le = self._encode_labels(data, le)
        logging.info("Label encoding complete")

        data['tenure'] = mms.fit_transform(data[['tenure']])
        data['MonthlyCharges'] = mms.fit_transform(data[['MonthlyCharges']])
        data['TotalCharges'] = mms.fit_transform(data[['TotalCharges']])
        logging.info("Data MinMax scaled")

        data.drop(columns = ['PhoneService', 'gender','StreamingTV','StreamingMovies','MultipleLines','InternetService'],inplace = True)
        logging.info("Insignificant columns dropped")

        if train:
            x, y = over.fit_resample(data.iloc[:, :13].values, data.iloc[:, 13].values)
            logging.info(f"Data resampled to balance classes. Counter: {Counter(y)}")
        else:
            x = data.iloc[:, :13].values
            y = data.iloc[:, 13].values
            
        return data, x, y, le

    def _encode_labels(self, data, le):
        logging.info('Label Encoder Transformation')
        text_data_features = [i for i in list(data.columns) if i not in list(data.describe().columns)]
        logging.info(text_data_features)
        for i in text_data_features :
            data[i] = le.fit_transform(data[i])
            logging.info(f"{i}:{data[i].unique()} = {le.inverse_transform(data[i].unique())}")
        
        return data, le

    def _cast_totalcharges_to_float(self, data):
        l1 = [len(i.split()) for i in data['TotalCharges']]
        l2 = [i for i in range(len(l1)) if l1[i] != 1]
        for i in l2:
            data.loc[i,'TotalCharges'] = data.loc[(i-1),'TotalCharges']
        data['TotalCharges'] = data['TotalCharges'].astype(float)
        return data

    def _push_features_to_store(self, df, feature_name):
        try:
            # Add timestamp column if not present
            if 'event_timestamp' not in df.columns:
                df['event_timestamp'] = pd.Timestamp.now()
            
            # Add entity_id column if not present
            if 'entity_id' not in df.columns:
                df['entity_id'] = range(len(df))

            # Save data as parquet
            data_path = os.path.join(
                self.data_transformation_config.feature_store_repo_path,
                "data"
            )
            parquet_path = os.path.join(data_path, f"{feature_name}_features.parquet")
            
            # Ensure the directory exists
            os.makedirs(data_path, exist_ok=True)
            
            # Save the parquet file
            df.to_parquet(parquet_path, index=False)
            logging.info(f"Saved feature data to {parquet_path}")

            # Define data source with relative path
            data_source = FileSource(
                path=f"data/{feature_name}_features.parquet",
                timestamp_field="event_timestamp"
            )

            # Define entity
            entity = Entity(
                name="entity_id",
                value_type=ValueType.INT64,
                description="Entity ID"
            )

            # Define feature view
            feature_view = FeatureView(
                name=f"{feature_name}_features",
                entities=[entity],
                schema=[
                    Field(name='SeniorCitizen',dtype=Int64),
                    Field(name='Partner',dtype=Int32),
                    Field(name='Dependents',dtype=Int32),
                    Field(name='tenure',dtype=Int64),
                    Field(name='OnlineSecurity',dtype=Int32),
                    Field(name='OnlineBackup',dtype=Int32),
                    Field(name='DeviceProtection',dtype=Int32),
                    Field(name='TechSupport',dtype=Int32),
                    Field(name='Contract',dtype=Int32),
                    Field(name='PaperlessBilling',dtype=Int32),
                    Field(name='PaymentMethod',dtype=Int32),
                    Field(name='MonthlyCharges',dtype=Float64),
                    Field(name='TotalCharges',dtype=Float64),
                    Field(name='Churn',dtype=Int32)
                ],
                source=data_source,
                online=True
            )

            # Apply to feature store
            self.feature_store.apply([entity, feature_view])
            logging.info(f"Applied entity and feature view for {feature_name}")

            # Materialize features
            self.feature_store.materialize(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now() + timedelta(days=1)
            )
            logging.info("Materialized features successfully")

        except Exception as e:
            logging.error("Error while pushing features to store", exc_info=e)
            exit(1)

    def _retrieve_features_from_store(self, feature_name, n_sample):
        try:
            feature_service_name = f"{feature_name}_features"
            feature_vector = self.feature_store.get_online_features(
                feature_refs=[
                    f"{feature_name}_features:SeniorCitizen",
                    f"{feature_name}_features:Partner",
                    f"{feature_name}_features:Dependents",
                    f"{feature_name}_features:tenure",
                    f"{feature_name}_features:OnlineSecurity",
                    f"{feature_name}_features:OnlineBackup",
                    f"{feature_name}_features:DeviceProtection",
                    f"{feature_name}_features:TechSupport",
                    f"{feature_name}_features:Contract",
                    f"{feature_name}_features:PaperlessBilling",
                    f"{feature_name}_features:PaymentMethod",
                    f"{feature_name}_features:MonthlyCharges",
                    f"{feature_name}_features:TotalCharges",
                    f"{feature_name}_features:Churn"
                ],
                entity_rows=[{"entity_id": i} for i in range(n_sample)]
            ).to_df()

            logging.info(f"Retrieved features for {feature_name}")
            return feature_vector

        except Exception as e:
            logging.error("Error occurred when retrieving features", exc_info=e)
            exit(1)