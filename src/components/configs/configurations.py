import os
from dataclasses import dataclass

ARTIFACT_FOLDER_PATH = "artifacts"
TRAIN_DATASET_FILE_NAME = "train.csv"
TEST_DATASET_FILE_NAME = "test.csv"
RAW_DATASET_FILE_NAME = "data.csv"
X_PREPROCESSOR_FILE_NAME = "x_preprocessor.pkl"
Y_PREPROCESSOR_FILE_NAME = "y_preprocessor.pkl"
TRAINED_MODEL_FILE_NAME = "model.pkl"

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACT_FOLDER_PATH, TRAIN_DATASET_FILE_NAME)
    test_data_path: str = os.path.join(ARTIFACT_FOLDER_PATH, TEST_DATASET_FILE_NAME)
    raw_data_path: str = os.path.join(ARTIFACT_FOLDER_PATH, RAW_DATASET_FILE_NAME)


@dataclass
class DataTransformationConfig:
    X_preprocessor_obj_file_path = os.path.join(ARTIFACT_FOLDER_PATH, X_PREPROCESSOR_FILE_NAME)
    Y_preprocessor_obj_file_path = os.path.join(ARTIFACT_FOLDER_PATH, Y_PREPROCESSOR_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(ARTIFACT_FOLDER_PATH, TRAINED_MODEL_FILE_NAME)
    thresholds = {"accuracy": 0.5, "f1_score": 0.5, "precision": 0.5, "recall": 0.5}



@dataclass
class PredictPipelineConfig:
    X_preprocessor_obj_file_path = os.path.join(ARTIFACT_FOLDER_PATH, X_PREPROCESSOR_FILE_NAME)
    Y_preprocessor_obj_file_path = os.path.join(ARTIFACT_FOLDER_PATH, Y_PREPROCESSOR_FILE_NAME)
    model_path = os.path.join(ARTIFACT_FOLDER_PATH,TRAINED_MODEL_FILE_NAME)

@dataclass
class WeightScoreConfig:
    accuracy_weight = 0.25
    f1_score_weight = 0.35
    precision_weight = 0.2
    recall_weight = 0.2
    