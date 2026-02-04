from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler config
with open("customer_churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_config.pkl", "rb") as f:
    config = pickle.load(f)

scaler = config["scaler"]
FEATURE_COLUMNS = config["feature_columns"]


def to_int(x, d=0):
    try:
        return int(x)
    except:
        return d


def to_float(x, d=0.0):
    try:
        return float(x)
    except:
        return d


def form_to_features(form):
    row = {
        # ===== NUMERIC (USE DATASET TENURE DIRECTLY) =====
        "tenure_yeo_trim": to_int(form.get("tenure")),
        "MonthlyCharges_yeo_trim": to_float(form.get("MonthlyCharges")),
        "TotalCharges_replaced_yeo_trim": to_float(form.get("TotalCharges")),

        # ===== BASIC BINARY =====
        "gender_Male": to_int(form.get("gender")),
        "SeniorCitizen": to_int(form.get("SeniorCitizen")),
        "Partner_Yes": to_int(form.get("Partner")),
        "Dependents_res": to_int(form.get("Dependents")),
        "PhoneService": to_int(form.get("PhoneService")),
        "PaperlessBilling_res": to_int(form.get("PaperlessBilling")),
        "Contract_res": to_int(form.get("Contract")),

        # ===== MULTIPLE LINES =====
        "MultipleLines_No phone service": 1 if form.get("MultipleLines") == "2" else 0,
        "MultipleLines_Yes": 1 if form.get("MultipleLines") == "1" else 0,

        # ===== INTERNET =====
        "InternetService_Fiber optic": 1 if form.get("InternetService") == "2" else 0,
        "InternetService_No": 1 if form.get("InternetService") == "0" else 0,

        # ===== SERVICES =====
        "OnlineSecurity_Yes": to_int(form.get("OnlineSecurity")),
        "OnlineBackup_Yes": to_int(form.get("OnlineBackup")),
        "DeviceProtection_Yes": to_int(form.get("DeviceProtection")),
        "TechSupport_Yes": to_int(form.get("TechSupport")),
        "StreamingTV_Yes": to_int(form.get("StreamingTV")),
        "StreamingMovies_Yes": to_int(form.get("StreamingMovies")),

        # ===== PAYMENT =====
        "PaymentMethod_Electronic check": 1 if form.get("PaymentMethod") == "0" else 0,
        "PaymentMethod_Mailed check": 1 if form.get("PaymentMethod") == "1" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if form.get("PaymentMethod") == "3" else 0,

        # ===== SIM =====
        "sim_Jio": 1 if form.get("sim") == "0" else 0,
        "sim_Vi": 1 if form.get("sim") == "2" else 0,
        "sim_BSNL": 0
    }

    return row


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = form_to_features(request.form)

        X = np.array([[row.get(col, 0) for col in FEATURE_COLUMNS]])
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]

        result = "❌ YES – Customer Will Churn" if pred == 1 else "✅ NO – Customer Will Stay"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
