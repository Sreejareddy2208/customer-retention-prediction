import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
import seaborn as sns
from log_file import setup_logging
logger=setup_logging('out_handle')
from scipy import stats
def trimming(x_train_num, x_test_num):
    try:
        original_cols = x_train_num.columns

        for i in original_cols:
            q1 = x_train_num[i].quantile(0.25)
            q3 = x_train_num[i].quantile(0.75)
            iqr = q3 - q1

            upper_limit = q3 + 1.5 * iqr
            lower_limit = q1 - 1.5 * iqr

            x_train_num[i + '_trim'] = np.where(
                x_train_num[i] > upper_limit, upper_limit,
                np.where(x_train_num[i] < lower_limit, lower_limit, x_train_num[i])
            )

            x_test_num[i + '_trim'] = np.where(
                x_test_num[i] > upper_limit, upper_limit,
                np.where(x_test_num[i] < lower_limit, lower_limit, x_test_num[i])
            )

        # ✅ Drop ALL original columns
        x_train_num = x_train_num.drop(columns=original_cols)
        x_test_num = x_test_num.drop(columns=original_cols)

        logger.info(f'x_train_num_cols {x_train_num.columns}')
        return x_train_num, x_test_num

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')
        raise



    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')