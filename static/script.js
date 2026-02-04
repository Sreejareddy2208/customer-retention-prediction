<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>

<div class="card">
    <h1>Customer Churn Prediction</h1>
    <p class="subtitle">Enter customer details to predict churn status</p>

    <div class="form-grid">
        <input type="number" id="tenure" placeholder="Tenure (months)">
        <input type="number" id="monthly" placeholder="Monthly Charges">
        <input type="number" id="total" placeholder="Total Charges">

        <select id="gender">
            <option value="">Gender</option>
            <option value="1">Male</option>
            <option value="0">Female</option>
        </select>

        <select id="partner">
            <option value="">Partner</option>
            <option value="1">Yes</option>
            <option value="0">No</option>
        </select>

        <select id="dependents">
            <option value="">Dependents</option>
            <option value="1">Yes</option>
            <option value="0">No</option>
        </select>
    </div>

    <button onclick="predict()">Predict</button>

    <div id="result" class="result-box"></div>
</div>

<script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>
