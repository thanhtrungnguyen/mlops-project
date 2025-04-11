# src/app.py
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Attempt to load the best model saved during training.
MODEL_PATH = "best_model"
try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print("Error loading model:", e)


# API endpoint for JSON-based predictions.
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    if "data" not in data:
        return jsonify({"error": "No data provided"}), 400

    features = data["data"]
    try:
        prediction = model.predict([features])[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# UI endpoint to render the HTML page.
@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


# UI endpoint to handle form submissions.
@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    features_str = request.form.get("features")
    # Convert the input string to a list of floats.
    try:
        features = [float(x.strip()) for x in features_str.split(",")]
    except Exception:
        return render_template("index.html", prediction="Invalid input. Please enter comma separated numbers.")

    try:
        prediction = model.predict([features])[0]
    except Exception as e:
        return render_template("index.html", prediction=f"Error during prediction: {e}")

    return render_template("index.html", prediction=int(prediction))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
