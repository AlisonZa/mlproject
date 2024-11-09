import os, sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
import joblib
from src.utils import evaluate_models

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname (__file__), "../.."))

@dataclass
class ModelTrainerConfig:
    # preprocessor output
    tuned_model_file_path: str = os.path.join(PROJECT_ROOT, "artifacts", "tuned_models")
    # tuned_model_reports: str = os.path.join(PROJECT_ROOT, "artifacts", "tuned_model_reports.txt")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        # TODO: Modularizar o x_train e y_train
        os.makedirs(self.model_trainer_config.tuned_model_file_path, exist_ok= True)
    
    def intiate_model_trainer_regression(self, 
                                        transformed_X_train,
                                        y_train, 
                                        model_list = [
                                            CatBoostRegressor(),
                                            AdaBoostRegressor(),
                                            GradientBoostingRegressor(), 
                                            RandomForestRegressor(),
                                            LinearRegression(),
                                            KNeighborsRegressor(),
                                            DecisionTreeRegressor(),
                                            XGBRegressor(),],
                                        metrics_list = [
                                            r2_score
                                        ]
                                        ):
        
        trained_model_list = []
        
        logging.info(f"Entered the {self.intiate_model_trainer_regression.__name__} method")
        
        try:
            logging.info(f"Starting the model training method")
            
            logging.info(f"Model_training_results:")
            
            for index, model in enumerate(model_list):
                model.fit(transformed_X_train, y_train)
                trained_model_list.append(model)
            
                logging.info(f"Trained: {model.__class__.__name__} | Index: {index}")
                
                y_pred = model.predict(transformed_X_train)

                for metric in metrics_list:
                    score = metric(y_train, y_pred)
                    logging.info(f"{metric.__name__}: {score}")

            logging.info(f"Sucessfully  ran the {self.intiate_model_trainer_regression.__name__} method\nCheck the logs to see the models performance")

            return trained_model_list 

        except Exception as e:
            
            logging.exception(e)
            raise CustomException(e, sys)
        
    def hyperparameter_tuning_grid_search(self, trained_models, transformed_X_train, y_train):
        """
        Method to perform hyperparameter tuning on user-selected models.
        
        Parameters:
        - trained_models: list of trained models to select from.
        - transformed_X_train: feature data used for training and tuning.
        - y_train: target data used for training and tuning.
        """
        try:
            # Prompt user to input indices of models for tuning
            model_indices = input("Enter the indices of models to tune, separated by commas (e.g., 0,3,5): ")
            selected_indices = [int(model_index.strip()) for model_index in model_indices.split(",")]

            # Dictionary to store tuned models and their best parameters
            tuned_models = {}

            for  model_index in selected_indices:
                model = trained_models[model_index]
                model_name = model.__class__.__name__
                print(f"\nSpecify the hyperparameter grid for {model_name} as a dictionary:")
                print('Example: {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}')

                # Loop until a valid hyperparameter grid dictionary is provided
                while True:
                    try:
                        # Prompt user to input the hyperparameter grid as a dictionary
                        param_grid = eval(input(f"Enter parameter grid for {model_name}: "))

                        # Check if param_grid is a dictionary
                        if not isinstance(param_grid, dict):
                            raise ValueError("Input must be a dictionary format.")

                        # Perform hyperparameter tuning using GridSearchCV
                        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                        grid_search.fit(transformed_X_train, y_train)

                        # Store the best estimator and parameters
                        tuned_models[model_name] = {
                            'best_estimator': grid_search.best_estimator_,
                            'best_params': grid_search.best_params_,
                            'best_score': grid_search.best_score_,
                        }

                        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                        logging.info(f"Best score for {model_name}: {grid_search.best_score_}")
                        break  # Exit the loop if successful

                    except (SyntaxError, NameError, ValueError, TypeError) as e:
                        print(f"Invalid input: {e}. Please enter the hyperparameter grid again as a dictionary.")
                        print('Example: {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}')

            logging.info("Hyperparameter tuning completed successfully.")
            return tuned_models

        except Exception as e:
            logging.exception("Exception occurred during hyperparameter tuning")
            raise CustomException(e, sys)
    

    def save_models(self, tuned_models):
        logging.info(f"Starting the process of saving the models:")
        try:
            # Ensure the output directory exists
            os.makedirs(self.model_trainer_config.tuned_model_file_path, exist_ok=True)

            # Prompt user to input indices of models for tuning
            model_names = input("Enter the name of models to export, separated by commas (e.g., 'CatBoostRegressor', 'LinearRegression'...): ")
            selected_models = [str(model_name.strip()) for model_name in model_names.split(",")]

            for model_name in selected_models:
                best_model = tuned_models[model_name]['best_estimator']
                
                # Set a unique file path for each model
                file_path = os.path.join(self.model_trainer_config.tuned_model_file_path, f"{model_name}_best_model.pkl")
                
                # Save the model using joblib
                joblib.dump(best_model, file_path) 

                logging.info(f"Succesfully saved {model_name} to: \n{file_path}")

        except Exception as e:
            logging.exception("Exception occurred during model_saving")
            raise CustomException(e, sys)


        
    def initiate_brute_force_approach(self,transformed_X_train, transformed_X_test, y_train, y_test):
        try:
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=transformed_X_train,y_train=y_train,X_test=transformed_X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # Set a unique file path for each model
            file_path = os.path.join(self.model_trainer_config.tuned_model_file_path, f"{(best_model_name).replace(' ', '_')}_best_model.pkl")
            
            # Save the model using joblib
            joblib.dump(best_model, file_path) 

            logging.info(f"Succesfully saved {best_model_name} to: \n{file_path}")

            predicted=best_model.predict(transformed_X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
           
        except Exception as e:
            raise CustomException(e,sys)



