from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and class names
with open("model.pkl", "rb") as f:
    model, class_names = pickle.load(f)

# Image mapping
flower_images = {
    "setosa": "setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}

# Care and climate information
flower_info = {
    "setosa": {
        "care": "• Moist, acidic soil\n• Partial shade to full sun\n• Water regularly",
        "climate": "Cool, wet climates like Alaska and Canada. Grows in boggy areas."
    },
    "versicolor": {
        "care": "• Wetland/garden soil\n• 6+ hours of sunlight\n• Divide rhizomes every 2–3 years",
        "climate": "Temperate and wet regions, handles cold winters and warm summers."
    },
    "virginica": {
        "care": "• Rich, moist soil\n• Thrives near ponds/swamps\n• Remove dead blooms",
        "climate": "Warm, humid subtropical climate (southeastern U.S.). Needs water & warmth."
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_file = None
    care = None
    climate = None

    if request.method == "POST":
        try:
            # Get 4 features from user input
            features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
            input_array = np.array(features).reshape(1, -1)

            # Predict flower type
            pred_probs = model.predict_proba(input_array)
            pred_class = np.argmax(pred_probs)
            prediction = class_names[pred_class]
            confidence = round(pred_probs[0][pred_class] * 100, 2)
            image_file = flower_images[prediction.lower()]
            care = flower_info[prediction.lower()]["care"]
            climate = flower_info[prediction.lower()]["climate"]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_file=image_file,
                           care_info=care,
                           climate_info=climate)

if __name__ == "__main__":
    app.run(debug=True)
