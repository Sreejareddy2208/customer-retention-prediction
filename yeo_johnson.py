import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
from log_file import setup_logging
logger=setup_logging('yeo_johnson')
from scipy import stats
def var_trans(x_train_num,x_test_num):
    try:
        logger.info(f"{x_train_num.columns} -> {x_train_num.shape}")

        original_cols = x_train_num.columns.tolist()

        for col in original_cols:
            x_train_num[col + '_yeo'], _ = stats.yeojohnson(x_train_num[col])
            x_test_num[col + '_yeo'], _ = stats.yeojohnson(x_test_num[col])

        # Drop only original columns
        x_train_num = x_train_num.drop(columns=original_cols)
        x_test_num = x_test_num.drop(columns=original_cols)

        return x_train_num, x_test_num

        # for i in x_train_num.columns:
        #     x_train_num[i + "_yeo"], lam = stats.yeojohnson(x_train_num[i])
        #     x_train_num = x_train_num.drop([i], axis=1)
        #
        #     iqr = x_train_num[i + "_yeo"].quantile(0.75) - x_train_num[i + "_yeo"].quantile(0.25)
        #     upper_limit = x_train_num[i + "_yeo"].quantile(0.75) + (1.5 * iqr)
        #     lower_limt = x_train_num[i + "_yeo"].quantile(0.25) - (1.5 * iqr)
        #     x_train_num[i + "_yeo_trim"] = np.where(x_train_num[i + "_yeo"] > upper_limit, upper_limit,
        #                                             np.where(x_train_num[i + "_yeo"] < lower_limt, lower_limt,
        #                                                      x_train_num[i + "_yeo"]))
        #     x_train_num = x_train_num.drop([i + "_yeo"], axis=1)
        #     x_test_num[i + "_yeo_trim"] = np.where(x_test_num[i] > upper_limit, upper_limit,
        #                                            np.where(x_test_num[i] < lower_limt, lower_limt,
        #                                                     x_test_num[i]))
        #     x_test_num = x_test_num.drop([i], axis=1)
        #     logger.info(f"{x_train_num.columns} -> {x_train_num.shape}")
        #     logger.info(f"{x_test_num.columns} -> {x_test_num.shape}")
        #
        # return x_train_num,x_test_num
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')