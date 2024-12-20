import os, sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

@dataclass
class DataTransformationConfig:
    # preprocessor output
    preprocessor_file_path: str = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    trained_preprocessor_file_path: str = os.path.join(PROJECT_ROOT, "artifacts", "trained_preprocessor.pkl")

    # Transformeds numpy arrays file paths
    transformed_X_train_file_path: str = os.path.join(PROJECT_ROOT, "artifacts", "transformed_X_train.npy")
    transformed_X_test_file_path: str = os.path.join(PROJECT_ROOT, "artifacts", "transformed_X_test.npy")
    y_train_file_path: str = os.path.join(PROJECT_ROOT, "artifacts", "y_train.npy")
    y_test_file_path: str = os.path.join(PROJECT_ROOT, "artifacts", "y_test.npy")


class DataTransformation:
    def __init__(self):
        self.transformation_config= DataTransformationConfig()

    def get_preprocessor_artifact(self):
        logging.info(f"Entered the {self.get_preprocessor_artifact.__name__} method")

        try:
            # Todo: passar isso daqui como uma lista, de yaml ou de classe para ser variável
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender",
                                   "race_ethnicity",
                                   "parental_level_of_education",
                                   "lunch",
                                   "test_preparation_course"]

            logging.info(f"Numerical Columns to preprocess: {numerical_columns}\nCategorical Columns to preprocess: {categorical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)), 
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            joblib.dump(preprocessor, self.transformation_config.preprocessor_file_path)

            logging.info(f"Saved the preprocessor artifact to:\n{self.transformation_config.preprocessor_file_path}")
            logging.info(f"Sucessfully ran the {self.get_preprocessor_artifact.__name__} method")


            return preprocessor

        except Exception as e:
            
            logging.exception(e)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train, test):
        logging.info(f"Entered the {self.initiate_data_transformation.__name__} method")

        try:
            preprocessor = joblib.load(self.transformation_config.preprocessor_file_path)
            logging.info(f"Successfully Loaded the preprocessor artifact")

            logging.info(f"Splitting the target from the features")
            target_column_name="math_score"
            
            X_train= train.drop(columns=[target_column_name],axis=1)
            y_train= train[target_column_name]

            X_test= test.drop(columns=[target_column_name],axis=1)
            y_test= test[target_column_name]
            logging.info(f"Succesfully splitted the the target from the features")

            logging.info(f"Training the preprocessor")
            trained_preprocessor = preprocessor.fit(X_train)
            joblib.dump(trained_preprocessor, self.transformation_config.trained_preprocessor_file_path)
            logging.info(f"Saved the trained_preprocessor artifact to:\n{self.transformation_config.trained_preprocessor_file_path}")

            logging.info(f"Starting Train_Data Preprocessing")
            transformed_X_train = trained_preprocessor.transform(X_train)
            logging.info(f"Successfully Ran rain_Data Preprocessing")

            logging.info(f"Starting Test Data_Preprocessing")
            transformed_X_test = trained_preprocessor.transform(X_test)
            logging.info(f"Successfully Ran Test_Data Preprocessing")

            logging.info(f"Saving files resulting numpy arrays to artifact folder")
            np.save(self.transformation_config.transformed_X_train_file_path, transformed_X_train)
            np.save(self.transformation_config.transformed_X_test_file_path, transformed_X_test)
            np.save(self.transformation_config.y_train_file_path, y_train)
            np.save(self.transformation_config.y_test_file_path, y_test)            

            logging.info(f"Saved the files to:\n{self.transformation_config.transformed_X_train_file_path}\n{self.transformation_config.transformed_X_test_file_path}\n{self.transformation_config.y_train_file_path}\n{self.transformation_config.y_test_file_path}")           
            logging.info(f"Sucessfully ran the {self.initiate_data_transformation.__name__} method")

            return transformed_X_train, transformed_X_test, y_train, y_test, trained_preprocessor

        except Exception as e:
            
            logging.exception(e)
            raise CustomException(e, sys)








