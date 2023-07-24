import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)


    except Exception as e:
        logging("Error in creating pickel")
        raise CustomException(e, sys)
    

def model_evaluation(X_train, X_test, y_train, y_test, models):
    try: 
        report = {}
        for i in range(len(list(models))):
            model  = list(models.values())[i]
            model.fit(X_train, y_train)


            ## Make predections
            y_pred = model.predict(X_test)
            
            ## Get r2 score for train and test data
            r2_scoure = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = r2_scoure

        
        return report


    except Exception as e:
        logging.info("Error occour in model eveluation!")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as load_obj:
            return pickle.load(load_obj)


    except Exception as e:
        logging.info('Error occoured in load object')
        raise CustomException(e, sys)