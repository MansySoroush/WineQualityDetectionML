import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.components.configs.configurations import ModelTrainerConfig

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = self.get_models_to_train()

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train,
                                                X_test = X_test, y_test = y_test,
                                                models = models)
            
            # To get best model accuracy from dict
            best_model_accuracy = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_accuracy)
            ]
            best_model = self.get_model_by_name(model_name= best_model_name, models= models)

            if best_model == None:
                raise CustomException("No best model found")

            if best_model_accuracy < self.model_trainer_config.model_threshold_accuracy:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"------------Results-----------")
            logging.info(f"Best found model name: {best_model_name}")
            logging.info(f"Best found model accuracy: {best_model_accuracy}")
            logging.info(f"------------------------------")


            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            return accuracy           
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_models_to_train(self):
        log_reg_params = {
            "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            "penalty": ['none', 'l1', 'l2', '‘elasticnet’'],
            "C": [100, 10, 1.0, 0.1, 0.01]
        }

        ridge_params = {
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }

        dec_tree_params = {
            'criterion':['gini','entropy', 'log_loss'],
            'splitter':['best','random'],
            'max_depth':[1,2,3,4,5],
            'max_features':['auto','sqrt','log2']
        }

        rf_params = {
            "max_depth": [5, 8, 15, None, 10],
            "max_features": ['auto', 'sqrt', 'log2'],
            "min_samples_split": [2, 8, 15, 20],
            "n_estimators": [100, 200, 500, 1000]
        }

        gradient_params={
            "loss": ['log_loss','deviance','exponential'],
            "criterion": ['friedman_mse','squared_error','mse'],
            "min_samples_split": [2, 8, 15, 20],
            "n_estimators": [100, 200, 500],
            "max_depth": [5, 8, 15, None, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        adaboost_params = {
            'n_estimators' : [50, 70, 90, 120, 180, 200],
            'learning_rate' : [0.001, 0.01, 0.1, 1, 10],
            'algorithm':['SAMME','SAMME.R']
            }

        xgboost_params = {
            "learning_rate": [0.1, 0.01],
            "max_depth": [5, 8, 12, 20, 30],
            "n_estimators": [100, 200, 300],
            "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4],
            "min_child_weight":[4,5,6],
            "reg_alpha":[0, 0.001, 0.005, 0.01, 0.05]
        }

        svc_params = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'degree': [2, 3, 4],
            'kernel': ['rbf','linear','poly','sigmoid']
        }

        cat_boost_params = {
            'learning_rate': [0.03, 0.06],
            'depth':[3, 6, 9],
            'l2_leaf_reg': [2, 3, 4],
            'boosting_type': ['Ordered', 'Plain']
        }

        knb_params = {
            'n_neighbors': range(1, 21, 2),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        models = [
            ("Logistic", LogisticRegression(), log_reg_params),
            ("Ridge", RidgeClassifier(), ridge_params),
            ("Decision Tree", DecisionTreeClassifier(), dec_tree_params),
            ("Random Forest", RandomForestClassifier(), rf_params),
            ("Gradient Boost", GradientBoostingClassifier(), gradient_params),
            ("Ada-boost", AdaBoostClassifier(), adaboost_params),
            ("Xgboost", XGBClassifier(), xgboost_params),
            ("SVC", SVC(kernel='linear'), svc_params),
            ("CatBoosting", CatBoostClassifier(), cat_boost_params),
            ("K-Neighbors", KNeighborsClassifier(), knb_params)
        ]

        return models

    def get_model_by_name(self, model_name, models):
        for name, model, params in models:
            if name == model_name:
                return model
            
        return None



if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    accuracy = model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(f"Accuracy Score of the Trained Model is: {accuracy}")
    

