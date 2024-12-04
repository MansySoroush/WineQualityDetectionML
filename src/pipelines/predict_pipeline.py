import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.configs.configurations import PredictPipelineConfig

class CustomData:
    def __init__(  self,        
        fixed_acidity: float,
        volatile_acidity: float,
        citric_acid: float,
        residual_sugar: float,
        chlorides: int,
        free_sulfur_dioxide: float,
        total_sulfur_dioxide: float,
        density: float,
        pH: float,
        sulphates: float,
        alcohol: float):

        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "fixed acidity": [self.fixed_acidity],
                "volatile acidity": [self.volatile_acidity],
                "citric acid": [self.citric_acid],
                "residual sugar": [self.residual_sugar],
                "chlorides": [self.chlorides],
                "free sulfur dioxide": [self.free_sulfur_dioxide],
                "total sulfur dioxide": [self.total_sulfur_dioxide],
                "density": [self.density],
                "pH": [self.pH],
                "sulphates": [self.sulphates],
                "alcohol": [self.alcohol],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    def __str__(self):
        return f"Physicochemical Tests( Fixed Acidity={self.fixed_acidity}, Volatile Acidity = {self.volatile_acidity}, Citric Acid = {self.citric_acid},\nResidual Sugar = {self.residual_sugar}, Chlorides = {self.chlorides}, Free Sulfur Dioxide = {self.free_sulfur_dioxide},\nTotal Sulfur Dioxide = {self.total_sulfur_dioxide}, Density = {self.density}, PH = {self.pH},\nSulphates = {self.sulphates}, Alcohol = {self.alcohol})"


class PredictPipeline:
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self,features):
        try:
            logging.info("Before Loading")

            model = load_object(file_path=self.predict_pipeline_config.model_path)
            x_preprocessor = load_object(file_path=self.predict_pipeline_config.X_preprocessor_obj_file_path)
            y_preprocessor = load_object(file_path=self.predict_pipeline_config.Y_preprocessor_obj_file_path)

            logging.info("After Loading")

            data_scaled = x_preprocessor.transform(features)
            y_pred = model.predict(data_scaled)

            logging.info(f"y_pred: {y_pred}")

            y_pred = np.round(y_pred).astype(int) 
            logging.info(f"y_pred (cast to int): {y_pred}")

            y_pred_original = y_preprocessor.inverse_transform(y_pred)
            
            return y_pred_original
        
        except Exception as e:
            raise CustomException(e,sys)
