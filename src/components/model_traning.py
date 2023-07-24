import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

## For handeling missing values we import simple imputer
## For numerical data mean, median, mode
## For categorical data we can use mode frequent
from sklearn.impute import SimpleImputer

## For feature scaling we used stander scaling
from sklearn.preprocessing import StandardScaler

## For Categorical order data  Ordinal Encoding 
from sklearn.preprocessing import OrdinalEncoder

### Pipeline
from sklearn.pipeline import Pipeline

## For combining above operation we need ColumnTransformer
from sklearn.compose import ColumnTransformer


## Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from src.utils import save_object, model_evaluation

@dataclass
class ModelTrainerConfig:
    trained_model_file_ptah = os.path.join('artifcats', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()


    def initate_model_traning(self, train_array, test_array):
        try:
            logging.info("Spliting dependent and independent features from train and test array")

            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]

            )

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet()

            }
            ## calling from utils will get the report after model train
            report:dict  = model_evaluation(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)
            print(report)
            print("\n", "="*100)
            logging.info(f'Model report: {report}')

            ## To get best model score from dictnory
            best_model_score = max(sorted(report.values()))

            ## best model name
            best_model_name = list(models.keys())[list(report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f'Best model found, Model name : {best_model_name}, R2 Score : {best_model_score} ')
            print("\n", "="*100)
            logging.info(f'Best model found, Model name : {best_model_name}, R2 Score : {best_model_score} ')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_ptah,
                obj=best_model
            )

        except Exception as e:
            logging.info("Error occoured in Model training")
            raise CustomException(e, sys)