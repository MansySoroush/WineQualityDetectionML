import sys
import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.components.configs.configurations import DataTransformationConfig
from src.utils import save_object

from src.components.data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation      
        '''
        try:
            # Initialize scaler
            scaler = StandardScaler()

            # Initialize the label encoder
            label_encoder = LabelEncoder()

            return scaler, label_encoder
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            x_preprocessing_obj,y_preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "quality"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data-frames.")

            input_feature_train_arr = x_preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = x_preprocessing_obj.transform(input_feature_test_df)

            target_feature_train_arr = y_preprocessing_obj.fit_transform(target_feature_train_df)
            target_feature_test_arr  = y_preprocessing_obj.transform(target_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_arr)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.X_preprocessor_obj_file_path,
                obj=x_preprocessing_obj
            )

            save_object(
                file_path=self.data_transformation_config.Y_preprocessor_obj_file_path,
                obj=y_preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.X_preprocessor_obj_file_path,
                self.data_transformation_config.Y_preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_,_ = data_transformation.initiate_data_transformation(train_data,test_data)
