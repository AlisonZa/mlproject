import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

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
            logging.info(f"Successfully ran the {self.initiate_data_ingestion.__name__} method")
            dataframe = pd.read_csv(self.ingestion_config.raw_data_path)
            
            # logging.info(f"Creating the folders to save the splits")

            # os.makedirs()

            # TODO stratify
            train_split, test_split = train_test_split(dataframe)

            train_split.to_csv(self.ingestion_config.train_data_path, index = False)
            test_split.to_csv(self.ingestion_config.test_data_path, index = False)

        except Exception as e:
            
            logging.exception(e)
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()


        








