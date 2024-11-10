import sys
import pandas as pd
from src.exception import CustomException
from dataclasses import dataclass
import joblib
import os
from src.logger import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join(PROJECT_ROOT, "artifacts", "tuned_models", "Linear_Regression_best_model.pkl")
        self.model = joblib.load(self.model_path)
        
        self.preprocessor_path = os.path.join(PROJECT_ROOT, "artifacts", "trained_preprocessor.pkl")
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(self, features):
        try:
            preprocessor = self.preprocessor
            model = self.model

            logging.info(f"Input data: {features}")
            transformed_input_data = preprocessor.transform(features)
            
            logging.info(f"Transformed data: {transformed_input_data}")
            predictions = model.predict(transformed_input_data)

            logging.info(f"Predicitons: {predictions}")
            return predictions
        except Exception as e:
            CustomException(e, sys)
             

class CustomData: #classe para converter os dados recebidos
    def __init__(self,
                gender:str,
                race_ethnicity:str,
                parental_level_of_education,
                lunch:str,
                test_preparation_course:str,
                reading_score: int,
                writing_score:int):
                 
                self.gender= gender
                self.race_ethnicity= race_ethnicity
                self.parental_level_of_education= parental_level_of_education
                self.lunch= lunch
                self.test_preparation_course= test_preparation_course
                self.reading_score= reading_score 
                self.writing_score= writing_score        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                 "gender": [self.gender],
                 "race_ethnicity": [self.race_ethnicity],
                 "parental_level_of_education": [self.parental_level_of_education],
                 "lunch": [self.lunch],
                 "test_preparation_course": [self.test_preparation_course],
                 "reading_score": [self.reading_score],
                 "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            CustomException(e, sys)
          


