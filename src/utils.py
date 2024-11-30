import os
import sys
import numpy as np
import pandas as pd
from src.exeception import CustomException
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train, X_test, y_train, y_test,models):
    try:
        report={}
        model_list=[]
        r2_list=[]
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            # print(f"FOR MACHINE LEARING MODEL:{model}\n")
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            # print(f"difference betwwen actual{y_train}\n and predicted {y_train_pred}\n =",y_train-y_train_pred)
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score



            print(list(models.keys())[i])
            model_list.append(list(models.keys())[i])
    
            print('Model performance for Training set')
            # print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
            # print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
            print("- R2 Score: {:.4f}".format(train_model_score))

            print('----------------------------------')
    
            print('Model performance for Test set')
            # print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
            # print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
            print("- R2 Score: {:.4f}".format(test_model_score))
            r2_list.append(test_model_score)
    
            print('='*35)
            print('\n')

        return report 
    except:
        pass
