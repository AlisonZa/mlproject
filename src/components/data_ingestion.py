import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(PROJECT_ROOT, "notebook", "data", "stud.csv")
    train_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "train.csv")
    test_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "test.csv") 

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"Entered the {self.initiate_data_ingestion.__name__} method")

        try:
            dataframe = pd.read_csv(self.ingestion_config.raw_data_path)

            # os.makedirs()

            # TODO stratify
            train_split, test_split = train_test_split(dataframe)

            train_split.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_split.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            
            logging.info(f"Successfully ran the {self.initiate_data_ingestion.__name__} method\nArtifacts saved to: {self.ingestion_config.train_data_path} \n{self.ingestion_config.test_data_path}")
            return train_split, test_split

        except Exception as e:
            
            logging.exception(e)
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # Data Ingestion
    data_ingestion_obj = DataIngestion()
    train_split, test_split = data_ingestion_obj.initiate_data_ingestion()
    
    # Data Transformation
    data_transformation_obj = DataTransformation()
    preprocessor = data_transformation_obj.get_preprocessor_artifact()
    transformed_X_train, transformed_X_test, y_train, y_test, trained_preprocessor = data_transformation_obj.initiate_data_transformation(train_split, test_split)
    
    # Model Training and hyperparam tuning:
    model_trainer_obj = ModelTrainer()
    # Minhas contribuições:
    ## trained_model_list = model_trainer_obj.intiate_model_trainer_regression(transformed_X_train, y_train)
    ## tuned_model_list = model_trainer_obj.hyperparameter_tuning_grid_search(trained_model_list, transformed_X_train, y_train)
    ## model_trainer_obj.save_models(tuned_model_list)
    
    r2_score = model_trainer_obj.initiate_brute_force_approach(transformed_X_train, transformed_X_test, y_train, y_test)
    











