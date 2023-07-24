import os
import sys
import numpy as np
import pandas as pd
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

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifcats', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self):
            try:
                logging.info("Data transformation inisated")
                ## Defining columns
                cat_col = ['cut', 'color', 'clarity']
                num_col = ['carat', 'depth', 'table', 'x', 'y', 'z'] 

                ## Defining custome ranking for categorical ordinal data
                cut_ranking = ['Fair', 'Good', 'Ideal', 'Very Good', 'Premium']
                color_ranking = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
                clarity_ranking = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

                logging.info("Pipeline inisited")

                ## Numerical pipelie
                num_pipeline = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]
                )

                ## Categorical pipeline
                cat_pipeline = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OrdinalEncoder(categories=[cut_ranking, color_ranking, clarity_ranking])),
                        ('scaler', StandardScaler())
                    ]
                )

                ## Preprossing
                preprocessor = ColumnTransformer([
                    ('num_pipeline', num_pipeline, num_col),
                    ('cat_pipeline', cat_pipeline, cat_col)
                ])

                logging.info("Pipeline completed")


                return preprocessor

            
            except Exception as e:
                 logging.info("Error in data transformation")
                 raise CustomException(e, sys)
            

    def initaite_data_transformation(self, train_path, test_path):
         try:
              logging.info('Reading Train and Test Data')

              train_df = pd.read_csv(train_path)
              test_df = pd.read_csv(test_path)

              logging.info('Read train and test data completed')
              logging.info(f'Train data head: \n {train_df.head().to_string()} ')
              logging.info(f'Test data head: \n {test_df.head().to_string()} ')


              logging.info('Obatning preprocessing object')
              preprocessing_obj = self.get_data_transformation_object()

              target_column_name = 'price'
              drop_columns = [target_column_name, 'id']

            ## Speliting into train and test data frame
              input_train_df = train_df.drop(columns=drop_columns, axis=1)
              target_train_df = train_df[target_column_name]

              input_test_df = test_df.drop(columns=drop_columns, axis=1 )
              target_test_df = test_df[target_column_name]

              logging.info('Applying preprocessor on traning and test dataset')

              ## Transforming using preprocessor object
              input_feature_train_arr = preprocessing_obj.fit_transform(input_train_df)
              input_feature_test_arr = preprocessing_obj.transform(input_test_df)

              ## Concatenating and converting into numpy arr 
              train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
              test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]


              save_object(
                   file_path=self.data_transformation_config.preprocessor_obj_file_path,
                   obj=preprocessing_obj
              )

              logging.info("Pickel file saved")

              return(
                   train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path
              )



         except Exception as e:
              logging.info("Error in iniaite data transformation")
              raise CustomException(e, sys)