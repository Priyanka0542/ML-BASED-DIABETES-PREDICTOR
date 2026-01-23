from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skinthickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        scaled_data = scaler.transform([data])
        result = model.predict(scaled_data)

        prediction = "Diabetic" if result[0] == 1 else "Non-Diabetic"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
