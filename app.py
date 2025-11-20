from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import joblib
import io
import base64

app = Flask(__name__)

cnn_model = load_model("mnist_cnn_model.h5")
svm_model = joblib.load("mnist_svm_model.pkl")

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize((28, 28))
    img_arr = np.array(img)
    img_arr = 255 - img_arr
    img_arr = img_arr / 255.0

    return img_arr

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["digit"].read()
    img_arr = preprocess(file)

    cnn_input = img_arr.reshape(1, 28, 28, 1)
    svm_input = img_arr.reshape(1, -1)

    # CNN prediction
    cnn_pred = cnn_model.predict(cnn_input)
    cnn_digit = int(np.argmax(cnn_pred))
    cnn_prob = float(np.max(cnn_pred)) * 100

    # SVM prediction + probability
    svm_proba = svm_model.predict_proba(svm_input)[0]
    svm_digit = int(np.argmax(svm_proba))
    svm_prob = float(np.max(svm_proba)) * 100

    # -------------------------------
    # FINAL COMPARISON LOGIC
    # -------------------------------
    if cnn_digit == svm_digit:
        best_model = "Both"
        final_digit = cnn_digit
        decision = f"Both models agree → Final Digit: {final_digit}"
    else:
        if cnn_prob >= svm_prob:
            best_model = "CNN"
            final_digit = cnn_digit
            decision = f"Models disagree → CNN chosen (Higher confidence)"
        else:
            best_model = "SVM"
            final_digit = svm_digit
            decision = f"Models disagree → SVM chosen (Higher confidence)"

    # Backend marks if chosen model is correct
    is_correct = (final_digit == cnn_digit) if best_model == "CNN" else (final_digit == svm_digit)

    # -------------------------------
    # DECISION TREE OUTPUT
    # -------------------------------
    tree_steps = [
        "Digit → Preprocess → CNN / SVM",
        f"CNN → Predicted: {cnn_digit} (Confidence: {round(cnn_prob, 2)}%)",
        f"SVM → Predicted: {svm_digit} (Confidence: {round(svm_prob, 2)}%)",
        f"Final → Best Model: {best_model}",
        f"Decision → {decision}"
    ]

    return jsonify({
        "cnn": {"digit": cnn_digit, "prob": round(cnn_prob, 2)},
        "svm": {"digit": svm_digit, "prob": round(svm_prob, 2)},
        "best_model": best_model,
        "final_digit": final_digit,
        "correct": is_correct,
        "tree": tree_steps
    })


if __name__ == "__main__":
    app.run(debug=True)


