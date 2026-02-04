import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from log_file import setup_logging
logger= setup_logging('random_sample_tect')
import warnings
warnings.filterwarnings('ignore')
def random_tech(x_train,x_test):
    try:
        logger.info(f'before filling null values training {x_train.shape}')
        logger.info(f'before filling null values testing {x_test.shape}')
        for i in x_train.columns:
            if x_train[i].isnull().sum()>0:
                x_train[i+'_replaced']=x_train[i].copy()
                x_test[i+'_replaced']=x_test[i].copy()
                s=x_train.dropna().sample(x_train[i].isnull().sum(),random_state=42)
                s1=x_test.dropna().sample(x_test[i].isnull().sum(),random_state=42)
                s.index=x_train[x_train[i].isnull()].index
                s1.index=x_test[x_test[i].isnull()].index
                x_train.loc[x_train[i].isnull(),i+'_replaced']=s
                x_test.loc[x_test[i].isnull(),i+'_replaced']=s1
                x_train=x_train.drop([i],axis=1)
                x_test=x_test.drop([i],axis=1)
        logger.info(f'after filling null values training {x_train.shape}')
        logger.info(f'after filling null values testing {x_test.shape}')
        logger.info(f'training data null values{x_train.isnull().sum()}')
        logger.info(f'testing data null values{x_test.isnull().sum()}')
        return x_train, x_test


    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')
