import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from pii_classification.inference.inference import NERPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for UI interaction

# Initialize predictor - path to the trained model directory
MODEL_PATH = os.path.abspath(
    "/mnt/data2/dev/develop/anonymizer-model/anon_model_output/20260416_130732/final_model"
)
predictor = NERPredictor(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        predictions = predictor.predict(text)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
