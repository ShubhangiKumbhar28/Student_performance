import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_mode_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],  #last column
                train_array[:,-1],   # last data y_train value
                test_array[:,:-1],  
                test_array[:,-1]

            )
            models = {
                
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()

            }

            params = {
                "Random Forest": {
                    'n_estimators': [6, 8, 16, 32, 64, 128]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'poisson', 'gini', 'entropy'],
                    # 'splitter': ['best', 'random'],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85],
                    'n_estimators': [6, 8, 16, 32, 64, 128]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 6, 7, 9],
                },
                "XGB Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [6, 8, 16, 32, 64, 128]
                },
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [6, 8, 16, 32, 64, 128]
                }
            }


            model_report:dict=evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, 
                                             y_test=y_test ,models=models, param=params)
            

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ##  To get best model score from dict
            best_model_name = list(model_report.keys())[

                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model founds")
            logging.info(f"Best model found on both train and test dataset is {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_mode_file_path,
                obj = best_model
            )

            predicted= best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)

