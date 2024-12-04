import os
import sys
import dill

from src.exception import CustomException
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
            # search_cv = GridSearchCV(estimator=model,
            #                         param_grid=params,
            #                         n_jobs=-1,
            #                         cv=3)
            search_cv = RandomizedSearchCV(estimator=model,
                                        param_distributions=params,
                                        n_iter=100,
                                        cv=3,
                                        verbose=2,
                                        random_state=42,
                                        n_jobs=-1)
            search_cv.fit(X_train, y_train)

            model.set_params(**search_cv.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Training set performance
            train_accuracy, train_f1, train_precision, train_recall = evaluate_model(y_train, y_train_pred)

            # Test set performance
            test_accuracy, test_f1, test_precision, test_recall = evaluate_model(y_test, y_test_pred)

            report[name] = test_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(true, predicted):
    accuracy = accuracy_score(true, predicted)  # Calculate Accuracy
    f1 = f1_score(true, predicted, average='weighted')  # Calculate F1-score for multi-class
    precision = precision_score(true, predicted, average='weighted')  # Calculate Precision for multi-class
    recall = recall_score(true, predicted, average='weighted')  # Calculate Recall for multi-class
    return accuracy, f1, precision, recall

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
