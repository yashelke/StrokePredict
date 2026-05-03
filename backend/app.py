# from flask import Flask, request, jsonify
# import pandas as pd
# from joblib import load
# from flask_cors import CORS


# model = load('stroke_prediction_model.joblib')


# app = Flask(__name__)
# CORS(app)


# @app.route('/predict', methods=['POST'])
# def predict():

#     try:
#         data = request.get_json()

#         df = pd.DataFrame([data])

#         prediction = model.predict(df)[0]

#         # return jsonify({"prediction": int(prediction[0])})

#         print("Prediction: ", {prediction})

#         return jsonify({"stroke": int(prediction)}), 200


#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/")

# def home():
#     return "<h1>Welcome to the Stroke Prediction App!</h1>"


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS
from flask import send_file

# from fpdf import fpdf
from io import BytesIO
import base64
import matplotlib.pyplot as plt

model = load("stroke_prediction_model.joblib")

app = Flask(__name__)

CORS(app)




@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        df = pd.DataFrame([data])

        prediction = int(model.predict(df)[0])

        probability = float(model.predict_proba(df)[0][1])

        probability_percent = round(probability * 100, 2)

        if probability < 0.3:
            risk_level = "Low"

        elif probability < 0.6:
            risk_level = "Moderate"

        else:
            risk_level = "High"

        risk_factors = []

        if float(data["age"]) > 60:
            risk_factors.append("High Age")

        if int(data["hypertension"]) == 1:
            risk_factors.append("Hypertension")

        if int(data["heart_disease"]) == 1:
            risk_factors.append("Heart Disease")

        if float(data["avg_glucose_level"]) > 140:
            risk_factors.append("High Glucose Level")

        if float(data["bmi"]) > 30:
            risk_factors.append("Obesity")

        if data["smoking_status"] in [
            # 'smokes',
            "formerly smoked"
        ]:
            risk_factors.append("Smoking History")

        if data["smoking_status"] == "smokes":
            risk_factors.append("Current Smoker")

        return (
            jsonify(
                {
                    "final_prediction": int(prediction),
                    "stroke": prediction,
                    "probability": probability_percent,
                    "risk_level": risk_level,
                    "risk_factors": risk_factors,
                }
            ),
            200,
        )

    except Exception as e:

        return jsonify({"error": str(e)}), 500








@app.route("/")
def home():
    return "<h1>StrokeSense - Welcome to Stroke Prediction App</h1>"


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
#
