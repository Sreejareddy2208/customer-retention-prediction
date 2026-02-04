import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from log_file import setup_logging

logger = setup_logging('cat_t_num')


def changing_data_to_num(x_train_cat, x_test_cat):
    try:
        # -------- 1. Binary Mapping --------
        binary_cols = ["PhoneService"]
        for col in binary_cols:
            x_train_cat[col] = x_train_cat[col].map({'Yes': 1, 'No': 0})
            x_test_cat[col] = x_test_cat[col].map({'Yes': 1, 'No': 0})

        # -------- 2. One-Hot Encoding --------
        onehot_cols = [
            "gender", "Partner", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "PaymentMethod", "sim"
        ]

        one_hot = OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        )

        one_hot.fit(x_train_cat[onehot_cols])

        # Train
        train_ohe = one_hot.transform(x_train_cat[onehot_cols])
        train_ohe_df = pd.DataFrame(
            train_ohe,
            columns=one_hot.get_feature_names_out(onehot_cols),
            index=x_train_cat.index
        )

        # Test
        test_ohe = one_hot.transform(x_test_cat[onehot_cols])
        test_ohe_df = pd.DataFrame(
            test_ohe,
            columns=one_hot.get_feature_names_out(onehot_cols),
            index=x_test_cat.index
        )

        # Drop original cols + concat
        x_train_cat = pd.concat([x_train_cat.drop(columns=onehot_cols), train_ohe_df], axis=1)
        x_test_cat = pd.concat([x_test_cat.drop(columns=onehot_cols), test_ohe_df], axis=1)

        # -------- 3. Ordinal Encoding --------
        ord_cols = ["Dependents", "PaperlessBilling", "Contract"]
        ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        ord_enc.fit(x_train_cat[ord_cols])

        # Train
        train_ord = ord_enc.transform(x_train_cat[ord_cols])
        train_ord_df = pd.DataFrame(
            train_ord,
            columns=[col + "_res" for col in ord_cols],
            index=x_train_cat.index
        )

        # Test
        test_ord = ord_enc.transform(x_test_cat[ord_cols])
        test_ord_df = pd.DataFrame(
            test_ord,
            columns=[col + "_res" for col in ord_cols],
            index=x_test_cat.index
        )

        x_train_cat = pd.concat([x_train_cat.drop(columns=ord_cols), train_ord_df], axis=1)
        x_test_cat = pd.concat([x_test_cat.drop(columns=ord_cols), test_ord_df], axis=1)

        # -------- 4. Logging --------
        logger.info(f"Train columns: {x_train_cat.columns}")
        logger.info(f"Test columns: {x_test_cat.columns}")
        logger.info(f"Train nulls:\n{x_train_cat.isnull().sum()}")
        logger.info(f"Test nulls:\n{x_test_cat.isnull().sum()}")

        return x_train_cat, x_test_cat

    except Exception as e:
        error_type, error_msg, tb = sys.exc_info()
        logger.error(f"Error in line {tb.tb_lineno}: {error_msg}")
        raise
