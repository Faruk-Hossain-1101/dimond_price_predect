import os
import sys
from src.logger import logging
from src.exception import CustomException


from src.components.data_ingestion import DataIngesation
from src.components.data_transformation import DataTransformation
from src.components.model_traning import ModelTrainer

if __name__ == "__main__":
    obj = DataIngesation()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initaite_data_transformation(train_path=train_data_path, test_path=test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initate_model_traning(train_array=train_arr, test_array=test_arr)