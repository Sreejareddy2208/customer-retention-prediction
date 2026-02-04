import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_file import setup_logging
logger = setup_logging('all_models')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
import pickle

def knn(x_train,y_train,x_test,y_test):
    try:
      knn_reg = KNeighborsClassifier(n_neighbors=5)
      knn_reg.fit(x_train,y_train)
      logger.info(f'KNN Test Accuracy : {accuracy_score(y_test,knn_reg.predict(x_test))}')
      logger.info(f'KNN Test confusion : {confusion_matrix(y_test, knn_reg.predict(x_test))}')
      logger.info(f'KNN classification report : {classification_report(y_test, knn_reg.predict(x_test))}')
      global knn_pred
      knn_pred = knn_reg.predict_proba(x_test)[:, 1]
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def nb(x_train,y_train,x_test,y_test):
    try:
      nb_reg = GaussianNB()
      nb_reg.fit(x_train,y_train)
      logger.info(f'Naive Bayes Test Accuracy : {accuracy_score(y_test,nb_reg.predict(x_test))}')
      logger.info(f'naive bayes Test confusion : {confusion_matrix(y_test, nb_reg.predict(x_test))}')
      logger.info(f'naive bayes classification report : {classification_report(y_test, nb_reg.predict(x_test))}')
      global nb_pred
      nb_pred = nb_reg.predict_proba(x_test)[:, 1]
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def lr(x_train,y_train,x_test,y_test):
    try:
      lr_reg = LogisticRegression()
      lr_reg.fit(x_train,y_train)
      logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test,lr_reg.predict(x_test))}')
      logger.info(f'LogisticRegression confusion matrix : {confusion_matrix(y_test, lr_reg.predict(x_test))}')
      logger.info(f'LogisticRegression  classification report {classification_report(y_test, lr_reg.predict(x_test))}')
      global lr_pred
      lr_pred = lr_reg.predict_proba(x_test)[:, 1]

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def dt(x_train,y_train,x_test,y_test):
    try:
      dt_reg = DecisionTreeClassifier(criterion='entropy')
      dt_reg.fit(x_train,y_train)
      logger.info(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(y_test,dt_reg.predict(x_test))}')
      logger.info(f'DecisionTreeClassifier confusion matrix : {confusion_matrix(y_test, dt_reg.predict(x_test))}')
      logger.info(f'DecisionTreeClassifier classification report : {classification_report(y_test, dt_reg.predict(x_test))}')
      global dt_pred
      dt_pred = dt_reg.predict_proba(x_test)[:, 1]
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def rf(x_train,y_train,x_test,y_test):
    try:
      rf_reg = RandomForestClassifier(n_estimators=5,criterion='entropy')
      rf_reg.fit(x_train,y_train)
      logger.info(f'RandomForestClassifier Test Accuracy : {accuracy_score(y_test,rf_reg.predict(x_test))}')
      logger.info(f'RandomForestClassifier confusion matrix  : {confusion_matrix(y_test, rf_reg.predict(x_test))}')
      logger.info(f'RandomForestClassifier classification report  : {classification_report(y_test, rf_reg.predict(x_test))}')
      global rf_pred
      rf_pred = rf_reg.predict_proba(x_test)[:, 1]
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def ada(x_train,y_train,x_test,y_test):
    try:
        t = LogisticRegression()
        ada_reg = AdaBoostClassifier(estimator=t,n_estimators=5)
        ada_reg.fit(x_train,y_train)
        logger.info(f'AdaBoostClassifier Test Accuracy : {accuracy_score(y_test, ada_reg.predict(x_test))}')
        logger.info(f'AdaBoostClassifier confusion matrix: {confusion_matrix(y_test, ada_reg.predict(x_test))}')
        logger.info(f'AdaBoostClassifier classification report: {classification_report(y_test, ada_reg.predict(x_test))}')
        global ada_pred
        ada_pred = ada_reg.predict_proba(x_test)[:, 1]
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def gb(x_train,y_train,x_test,y_test):
    try:
        gb_reg = GradientBoostingClassifier(n_estimators=5)
        gb_reg.fit(x_train,y_train)
        logger.info(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(y_test, gb_reg.predict(x_test))}')
        logger.info(f'GradientBoostingClassifier confusion matrix : {confusion_matrix(y_test, gb_reg.predict(x_test))}')
        logger.info(f'GradientBoostingClassifier classification report : {classification_report(y_test, gb_reg.predict(x_test))}')
        global gb_pred
        gb_pred = gb_reg.predict_proba(x_test)[:, 1]
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def xgb(x_train,y_train,x_test,y_test):
    try:
        xg_reg = XGBClassifier()
        xg_reg.fit(x_train, y_train)
        logger.info(f'XGBClassifier Test Accuracy : {accuracy_score(y_test,xg_reg.predict(x_test))}')
        logger.info(f'XGBClassifier confusion matrix: {confusion_matrix(y_test, xg_reg.predict(x_test))}')
        logger.info(f'XGBClassifier classification report : {classification_report(y_test, xg_reg.predict(x_test))}')
        global xg_pred
        xg_pred = xg_reg.predict_proba(x_test)[:, 1]
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def svm_c(x_train,y_train,x_test,y_test):
    try:
        svm_reg = SVC(kernel='rbf',probability=True)
        svm_reg.fit(x_train,y_train)
        logger.info(f'SVM Test Accuracy : {accuracy_score(y_test, svm_reg.predict(x_test))}')
        logger.info(f'SVM confusion matrix : {confusion_matrix(y_test, svm_reg.predict(x_test))}')
        logger.info(f'SVM classification report : {classification_report(y_test, svm_reg.predict(x_test))}')
        global svm_pred
        svm_pred = svm_reg.predict_proba(x_test)[:, 1]

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
def auc_roc(x_train,y_train,x_test,y_test):
    try:
        knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_pred)
        nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_pred)  # ✅ fixed
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred)
        dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_pred)
        rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred)
        ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_pred)
        gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_pred)
        xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xg_pred)
        svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_pred)

        plt.figure(figsize=(8, 5))
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(knn_fpr, knn_tpr, label="KNN")
        plt.plot(nb_fpr, nb_tpr, label="NB")
        plt.plot(lr_fpr, lr_tpr, label="LR")
        plt.plot(dt_fpr, dt_tpr, label="DT")
        plt.plot(rf_fpr, rf_tpr, label="RF")
        plt.plot(ada_fpr, ada_tpr, label="ADA")
        plt.plot(gb_fpr, gb_tpr, label="GB")
        plt.plot(xgb_fpr, xgb_tpr, label="XGB")
        plt.plot(svm_fpr, svm_tpr, label="SVM")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - All Models")
        plt.legend()
        plt.show()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


def common(x_train,y_train,x_test,y_test):
    try:

        logger.info('--KNN Algorithm--')
        knn(x_train,y_train,x_test,y_test)
        logger.info('--Naive Bayes Algorithm--')
        nb(x_train,y_train,x_test,y_test)
        logger.info('--Logistic Regression Algorithm--')
        lr(x_train,y_train,x_test,y_test)
        logger.info('--Decision Tree Algorithm--')
        dt(x_train,y_train,x_test,y_test)
        logger.info('--Random Forest Algorithm--')
        rf(x_train,y_train,x_test,y_test)
        logger.info('--Ada Boost Algorithm--')
        ada(x_train,y_train,x_test,y_test)
        logger.info('--Gradient Boosting Algorithm--')
        gb(x_train,y_train,x_test,y_test)
        logger.info('--XGBoost Algorithm--')
        xgb(x_train,y_train,x_test,y_test)
        logger.info('--SVM Algorithm--')
        svm_c(x_train, y_train, x_test, y_test)




        logger.info(f"KNN AUC  : {roc_auc_score(y_test, knn_pred)}")
        logger.info(f"NB AUC   : {roc_auc_score(y_test, nb_pred)}")
        logger.info(f"LR AUC   : {roc_auc_score(y_test, lr_pred)}")
        logger.info(f"DT AUC   : {roc_auc_score(y_test, dt_pred)}")
        logger.info(f"RF AUC   : {roc_auc_score(y_test, rf_pred)}")
        logger.info(f"ADA AUC  : {roc_auc_score(y_test, ada_pred)}")
        logger.info(f"GB AUC   : {roc_auc_score(y_test, gb_pred)}")
        logger.info(f"XGB AUC  : {roc_auc_score(y_test, xg_pred)}")
        logger.info(f"SVM AUC  : {roc_auc_score(y_test, svm_pred)}")
        auc_roc(x_train, y_train, x_test, y_test)


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')