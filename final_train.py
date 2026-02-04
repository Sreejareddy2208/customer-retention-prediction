import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')
from log_file import setup_logging
logger = setup_logging('final_train')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
def lr(x_train,y_train,x_test,y_test):
    try:
        reg=LogisticRegression()
        reg=LogisticRegression()
        reg.fit(x_train,y_train)
        logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test, reg.predict(x_test))}')
        logger.info(f'LogisticRegression confusion matrix : {confusion_matrix(y_test, reg.predict(x_test))}')
        logger.info(f'LogisticRegression  classification report {classification_report(y_test, reg.predict(x_test))}')
        with open('customer_churn.pkl','wb') as f:
            pickle.dump(reg,f)
        with open ('customer_churn.pkl','rb') as p:
            model=pickle.load(p)

        with open('scalar.pkl','rb') as p1:
            scal=pickle.load(p1)
        random_inputs = np.random.random((1, x_train.shape[1]))
        random_inputs_scaled = scal.transform(random_inputs)
        result_from_model = model.predict(random_inputs_scaled)
        logger.info(f'model prediction{result_from_model}')
        if result_from_model[0]==0:
            logger.info(f'bad customer')
        else:
            logger.info(f'good customer')






    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')