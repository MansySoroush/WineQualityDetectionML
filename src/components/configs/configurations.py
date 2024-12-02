import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")


@dataclass
class DataTransformationConfig:
    X_preprocessor_obj_file_path = os.path.join('artifacts',"x_preprocessor.pkl")
    Y_preprocessor_obj_file_path = os.path.join('artifacts',"y_preprocessor.pkl")
