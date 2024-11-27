from flask import Flask, request, jsonify
from src.pipelines.prediction_pipeline import CustomerChurnPredictionPipeline, CustomerData
import numpy as np
from src.logger.basic_logging import logging

app = Flask(__name__)

prediction_pipeline = CustomerChurnPredictionPipeline()

@app.route("/reload", methods=["GET"])
def reload():
    prediction_pipeline.load_artifacts()
    return "Success"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        json_data = request.get_json()
        logging.info(f"INCOMING_PARAMS: {json_data}")
        
        # Create CustomClass instance with JSON data
        data = CustomerData(
            customer_id=str(json_data.get('customerID')),
            gender=str(json_data.get('gender')),
            senior_citizen=int(json_data.get('SeniorCitizen')),
            partner=str(json_data.get('Partner')),
            dependents=str(json_data.get('Dependents')),
            tenure=int(json_data.get('tenure')),
            phone_service=str(json_data.get('PhoneService')),
            multiple_lines=str(json_data.get('MultipleLines')),
            internet_service=str(json_data.get('InternetService')),
            online_security=str(json_data.get('OnlineSecurity')),
            online_backup=str(json_data.get('OnlineBackup')),
            device_protection=str(json_data.get('DeviceProtection')),
            tech_support=str(json_data.get('TechSupport')),
            streaming_tv=str(json_data.get('StreamingTV')),
            streaming_movies=str(json_data.get('StreamingMovies')),
            contract=str(json_data.get('Contract')),
            paperless_billing=str(json_data.get('PaperlessBilling')),
            payment_method=str(json_data.get('PaymentMethod')),
            monthly_charges=float(json_data.get('MonthlyCharges')),
            total_charges=str(json_data.get('TotalCharges'))


        )

        # Get prediction
        final_data = data.get_pd_dataframe()
        pred = prediction_pipeline.predict(final_data)

        # Return prediction result
        return jsonify({
            "status": "success",
            "churn_raw_prediction": int(pred[0]),
            "churn_prediction": "Yes" if pred[0] == 1 else "No"
        })

    except Exception as e:
        logging.error("Churn prediction failed", exc_info=e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port="8000", debug=True)
    except Exception as e:
        logging.error("Application crashed", exc_info=e)
        exit(1)