"""
Feature selection. To reduce 31 features to 21, call all_selections with combined data after cat_to_num:
    training_data, testing_data = all_selections(x_num, x_test_num, y_train, training_data, testing_data)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import seaborn as sns
import logging
from log_file import setup_logging
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

logger = setup_logging('f_selection')
TARGET_FEATURES = 18  # Reduce to 18 inputs for model


def select_top_n_features(x_train, x_test, y_train, n=21):
    """Select top n features by absolute Pearson correlation with target. Use when data has > n columns."""
    if x_train.shape[1] <= n:
        return x_train, x_test
    # Target safe conversion
    if y_train.dtype == 'object':
        y_train = y_train.str.strip().str.capitalize().map({'Yes': 1, 'No': 0})
    if y_train.isnull().any():
        y_train = y_train.fillna(y_train.mode()[0])
    y_train = y_train.astype(int)
    # Pearson correlation
    corrs = []
    for col in x_train.columns:
        r, _ = pearsonr(x_train[col], y_train)
        corrs.append((col, abs(r)))
    corrs.sort(key=lambda x: x[1], reverse=True)
    top_cols = [c[0] for c in corrs[:n]]
    logger.info(f"Selected top {n} features: {top_cols}")
    return x_train[top_cols], x_test.reindex(columns=top_cols, fill_value=0)


def all_selections(x_train_num, x_test_num, y_train, x_train_combined=None, x_test_combined=None):
    """
    Feature selection. If x_train_combined/x_test_combined provided with >21 cols, reduces to 21.
    Otherwise runs variance-based selection on numeric data.
    """
    # When combined data (31 cols) is passed, reduce to 21 features
    if (x_train_combined is not None and x_test_combined is not None and
            x_train_combined.shape[1] > TARGET_FEATURES):
        logger.info(f"Reducing {x_train_combined.shape[1]} features to {TARGET_FEATURES}")
        return select_top_n_features(x_train_combined, x_test_combined, y_train, TARGET_FEATURES)

    try:
        logger.info(f" before constant{x_train_num.columns} -> {x_train_num.shape}")
        # constant:
        train_index = x_train_num.index
        test_index = x_test_num.index
        reg_con=VarianceThreshold(0.0)
        reg_con.fit(x_train_num)
        best_cols=x_train_num.columns[reg_con.get_support()]
        x_train_num=reg_con.transform(x_train_num)
        x_test_num = reg_con.transform(x_test_num)
        x_train_num = pd.DataFrame(x_train_num, columns=best_cols, index=train_index)
        x_test_num = pd.DataFrame(x_test_num, columns=best_cols, index=test_index)

        logger.info(f" after constant{x_train_num.columns} -> {x_train_num.shape}")
        logger.info(f'==================================================')
        logger.info(f" before quasi{x_train_num.columns} -> {x_train_num.shape}")
        # quasi:
        reg_quasi = VarianceThreshold(0.1)
        reg_quasi.fit(x_train_num)
        best_cols_quasi = x_train_num.columns[reg_quasi.get_support()]
        x_train_num = reg_quasi.transform(x_train_num)
        x_test_num = reg_quasi.transform(x_test_num)
        x_train_num = pd.DataFrame(x_train_num, columns=best_cols_quasi, index=train_index)
        x_test_num = pd.DataFrame(x_test_num, columns=best_cols_quasi, index=test_index)
        logger.info(f" after quasi{x_train_num.columns} -> {x_train_num.shape}")
        logger.info('================hypothesis=========')



        # ----- TARGET SAFE CONVERSION -----
        if y_train.dtype == 'object':
            y_train = (y_train.str.strip().str.capitalize().map({'Yes': 1, 'No': 0}))
        if y_train.isnull().any():
            y_train = y_train.fillna(y_train.mode()[0])
        y_train = y_train.astype(int)
        logger.info(f'{y_train.unique()}')
        values = []
        # plt.figure(figsize=(5, 3))
        for i in x_train_num.columns:
            values.append(pearsonr(x_train_num[i], y_train))
        values = np.array(values)
        p_values = pd.Series(values[:, 1], index=x_train_num.columns)
        p_values.sort_values(ascending=False, inplace=True)
        logger.info(f"{x_train_num.columns} -> {x_train_num.shape}")
        logger.info(f"{x_test_num.columns} -> {x_test_num.shape}")
        return x_train_num, x_test_num

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f'Issue at line {er_lin.tb_lineno} : {er_msg}')
