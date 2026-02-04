import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import warnings
from cat_t_num import changing_data_to_num
warnings.filterwarnings('ignore')
from log_file import setup_logging
logger = setup_logging('main')
from sklearn.model_selection import train_test_split
from random_sample_tect import random_tech
from yeo_johnson import var_trans
from out_handle import trimming
from f_selection import all_selections
from cat_t_num import changing_data_to_num
from balancing import balanc_data
from all_models import common
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
class CUSTOMER_CHURN():
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(self.df.isnull().sum())
            logger.info(self.df.shape)
            logger.info(self.df.dtypes)
            self.df=self.df.drop(columns=['customerID'])
            self.df['TotalCharges']=pd.to_numeric(self.df['TotalCharges'],errors='coerce')
            logger.info(self.df.isnull().sum())
            self.x=self.df.iloc[:,:-1]#independent
            self.y=self.df.iloc[:,-1]#dependent
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            logger.info(f'training data {self.x_train.shape}')
            logger.info(f'test data {self.x_test.shape}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')
    def missing_values(self):
        try:
            self.x_train, self.x_test = random_tech(self.x_train, self.x_test)
            logger.info('missing values handled')
            self.x_train_num = self.x_train.select_dtypes(exclude='object')
            self.x_train_cat = self.x_train.select_dtypes(include='object')
            self.x_test_num = self.x_test.select_dtypes(exclude='object')
            self.x_test_cat = self.x_test.select_dtypes(include='object')
            logger.info(f'Numeric train columns: {self.x_train_num.columns}')
            logger.info(f'Categorical train columns: {self.x_train_cat.columns}')
            logger.info(f'Numeric test columns: {self.x_test_num.columns}')
            logger.info(f'Categorical test columns: {self.x_test_cat.columns}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')
    def variable_transform(self):
        try:
            self.x_train_num,self.x_test_num=var_trans(self.x_train_num,self.x_test_num)
            logger.info(f' var trans {self.x_train_num.isnull().sum()}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')
    def outlier_handle(self):
        try:
            logger.info(f"Before  outliers: {self.x_train_num.columns} -> {self.x_train_num.shape}")
            logger.info(f"Before  outliers: {self.x_test_num.columns} -> {self.x_test_num.shape}")
            self.x_train_num,self.x_test_num=trimming(self.x_train_num,self.x_test_num)
            logger.info(f"After  outliers: {self.x_train_num.columns} -> {self.x_train_num.shape}")
            logger.info(f"After outliers: {self.x_test_num.columns} -> {self.x_test_num.shape}")
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')
    def feature_Selection(self):
        try:
            logger.info(f"Before feature selection: {self.x_train_num.columns} -> {self.x_train_num.shape}")
            logger.info(f"Before  feature selection: {self.x_test_num.columns} -> {self.x_test_num.shape}")
            self.x_train_num, self.x_test_num = all_selections(self.x_train_num, self.x_test_num,self.y_train)
            logger.info(f"After feature selection : {self.x_train_num.columns} -> {self.x_train_num.shape}")
            logger.info(f"After  feature selection: {self.x_test_num.columns} -> {self.x_test_num.shape}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')
    def cat_to_num(self):
        try:
            logger.info(f' before cat_to_num{self.x_train_cat.columns}')
            logger.info(f'before cat_to_num{self.x_test_cat.columns}')
            self.x_train_cat,self.x_test_cat=changing_data_to_num(self.x_train_cat,self.x_test_cat)
            logger.info(f' after cat_to_num{self.x_train_cat.columns}')
            logger.info(f'after cat_to_num{self.x_test_cat.columns}')
            self.x_train_num.reset_index(drop=True,inplace=True)
            self.x_train_cat.reset_index(drop=True,inplace=True)
            self.x_test_num.reset_index(drop=True,inplace=True)
            self.x_test_cat.reset_index(drop=True,inplace=True)
            self.training_data=pd.concat([self.x_train_num,self.x_train_cat],axis=1)
            self.testing_data = pd.concat([self.x_test_num, self.x_test_cat],axis=1)
            # Reduce to 18 features
            self.training_data, self.testing_data = all_selections(
                self.x_train_num, self.x_test_num, self.y_train,
                self.training_data, self.testing_data
            )
            logger.info(f'final training data')
            logger.info(f'{self.training_data.columns}')
            logger.info(f'{self.training_data.sample(10)}')
            logger.info(f'{self.training_data.isnull().sum()}')
            logger.info(f'final testing data')
            logger.info(f'{self.testing_data.columns}')
            logger.info(f'{self.testing_data.sample(10)}')
            logger.info(f'{self.testing_data.isnull().sum()}')
            logger.info(f'{self.training_data.shape}')
            logger.info(f'{self.testing_data.shape}')
            logger.info(self.training_data.dtypes)
            logger.info(
                self.training_data.select_dtypes(include='object').head()
            )

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')

    def data_balancing(self):
        try:
            def safe_target_mapping(y):
                if y.dtype == 'object':
                    y = y.map({'Yes': 1, 'No': 0})
                return y.astype('int')  # nullable integer type

            self.y_train = safe_target_mapping(self.y_train)
            self.y_test = safe_target_mapping(self.y_test)

            balanc_data(self.training_data, self.y_train,self.testing_data,self.y_test)


        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no : {error_line.tb_lineno} : due to {error_msg}')


if __name__ == "__main__":
    try:
        obj = CUSTOMER_CHURN(
            r"C:\Users\SREEJA REDDY\OneDrive\Attachments\Desktop\customer churn\churn prediction edited.csv"
        )
        obj.missing_values()
        obj.variable_transform()
        obj.outlier_handle()
        obj.feature_Selection()
        obj.cat_to_num()
        obj.data_balancing()

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')

