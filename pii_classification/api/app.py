import os

from flask_cors import CORS
from flask import Flask, request, jsonify

from pii_classification.inference.inference import AnonPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for UI interaction

# ----------------------------------------------------------------------
# Model registry – map a friendly name to the absolute path of the model
# ----------------------------------------------------------------------
MODEL_PATHS = {
    "1-PLC: 20260417_213751": "/mnt/data2/dev/develop/anonymizer-model/anon_model_output/20260417_213751/final_model",
    "2-PLC: 20260416_130732": "/mnt/data2/dev/develop/anonymizer-model/anon_model_output/20260416_130732/final_model",
}

USE_QUANTIZATION = True

# Load a predictor for every model at startup
PREDICTORS = {
    name: AnonPredictor(
        model_path=os.path.abspath(path), use_quantized=USE_QUANTIZATION
    )
    for name, path in MODEL_PATHS.items()
}

# Choose a default model – the first one in the dict
DEFAULT_MODEL = next(iter(PREDICTORS))


# ----------------------------------------------------------------------
# API: return the list of available model names
# ----------------------------------------------------------------------
@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": list(PREDICTORS.keys()), "default": DEFAULT_MODEL})


# ----------------------------------------------------------------------
# API: predict – optionally receive the model name to use
# ----------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text = data.get("text", "")
    model_name = data.get("model", DEFAULT_MODEL)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if model_name not in PREDICTORS:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

    predictor = PREDICTORS[model_name]

    try:
        predictions = predictor.predict(
            text=text, clean_punct=True, merge_entities=True, handle_gaps=True
        )
        return jsonify({"model": model_name, "predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
