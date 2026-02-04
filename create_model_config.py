"""
Create model_config.pkl with scaler + 18 feature columns.
Run this once if you get "X has 31 features, but StandardScaler is expecting 18".
Uses the same pipeline as main.py to determine the 18 columns.
"""
import pickle
import os

# Run from project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main import CUSTOMER_CHURN

csv_path = os.path.join(os.path.dirname(__file__), "churn prediction edited.csv")

obj = CUSTOMER_CHURN(csv_path)
obj.missing_values()
obj.variable_transform()
obj.outlier_handle()
obj.feature_Selection()
obj.cat_to_num()

# training_data now has 18 columns
feature_columns = list(obj.training_data.columns)
print(f"18 feature columns: {feature_columns}")

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_config = {'scaler': scaler, 'feature_columns': feature_columns}
with open('model_config.pkl', 'wb') as f:
    pickle.dump(model_config, f)

print("model_config.pkl created successfully.")
