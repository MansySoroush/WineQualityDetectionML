import os
import sys
import dill

from src.exception import CustomException
from src.components.configs.configurations import WeightScoreConfig
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model, params in models:
            search_cv = RandomizedSearchCV(estimator=model,
                                        param_distributions=params,
                                        n_iter=100,
                                        cv=3,
                                        verbose=2,
                                        random_state=42,
                                        n_jobs=-1)
            search_cv.fit(X_train, y_train)

            # Update model with best parameters
            model.set_params(**search_cv.best_params_)
            model.fit(X_train,y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate performance (Training set & Test set)
            train_metrics = evaluate_model(y_train, y_train_pred)
            test_metrics = evaluate_model(y_test, y_test_pred)

            # Add to report
            report[name] = {
                "params": search_cv.best_params_,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(true, predicted):
    metrics = {
        "accuracy": accuracy_score(true, predicted),
        "f1_score": f1_score(true, predicted, average='weighted'),
        "precision": precision_score(true, predicted, average='weighted'),
        "recall": recall_score(true, predicted, average='weighted'),
    }
    return metrics

def calculate_score(metrics, weights=None):
    if weights is None:
        weights = { 
            "accuracy": WeightScoreConfig.accuracy_weight, 
            "f1_score": WeightScoreConfig.f1_score_weight, 
            "precision": WeightScoreConfig.precision_weight, 
            "recall": WeightScoreConfig.recall_weight
        }
    
    score = 0
    for metric, weight in weights.items():
        score += metrics[metric] * weight
    return score

def meets_thresholds(metrics, thresholds):
    return all(metrics[metric] >= threshold for metric, threshold in thresholds.items())

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
