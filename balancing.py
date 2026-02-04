from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
import sys
from log_file import setup_logging
from final_train import lr

logger = setup_logging('balancing')


def balanc_data(X_train, y_train, X_test, y_test):
    try:
        # ---------- 1. Safety check ----------
        logger.info("Checking column consistency...")
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # ---------- 2. Before SMOTE ----------
        logger.info(f'Before Good class: {sum(y_train == 1)}')
        logger.info(f'Before Bad  class: {sum(y_train == 0)}')

        # ---------- 3. SMOTE (TRAIN ONLY) ----------
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

        logger.info(f'After Good class: {sum(y_train_bal == 1)}')
        logger.info(f'After Bad  class: {sum(y_train_bal == 0)}')

        # ---------- 4. Scaling ----------
        scaler = StandardScaler()
        X_train_bal_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)

        # ---------- 5. Save scaler + feature columns (for app prediction) ----------
        model_config = {
            'scaler': scaler,
            'feature_columns': list(X_train.columns)
        }
        with open('model_config.pkl', 'wb') as f:
            pickle.dump(model_config, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # ---------- 6. Model training ----------
        lr(X_train_bal_scaled, y_train_bal, X_test_scaled, y_test)

    except Exception as e:
        error_type, error_msg, tb = sys.exc_info()
        logger.error(f'Error in line {tb.tb_lineno}: {error_msg}')
        raise
