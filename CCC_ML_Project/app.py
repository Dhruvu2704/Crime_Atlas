from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# =========================
# Load Model & Scaler
# =========================
kmeans = joblib.load("model/kmeans.pkl")
scaler = joblib.load("model/scaler.pkl")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("static/data/city_crime_clusters.csv")

cluster_names = {
    0: "High Crime Metropolitan Cities",
    1: "Moderate Crime Cities",
    2: "Low Crime Cities",
    3: "Outlier Cities"
}

# =========================
# Routes
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/clusters")
def clusters():
    return render_template("clusters.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/about")
def about():
    return render_template("about.html")


# =========================
# Prediction API
# =========================

@app.route("/predict_cluster", methods=["POST"])
def predict_cluster():

    try:
        data = request.json

        input_data = np.array([[
            float(data["total_crimes"]),
            float(data["victim_age"]),
            float(data["police_deployed"]),
            float(data["weapon_rate"]),
            float(data["closure_rate"])
        ]])

        # Scale input
        scaled = scaler.transform(input_data)

        # Predict cluster
        cluster = int(kmeans.predict(scaled)[0])

        # Distance from nearest centroid
        distances = kmeans.transform(scaled)
        closest_distance = float(distances.min())

        # Confidence Score
        confidence = max(
            50,
            min(
                99,
                int(100 - (closest_distance * 25))
            )
        )

        return jsonify({
            "success": True,
            "cluster": cluster,
            "cluster_name": cluster_names[cluster],
            "confidence": confidence,
            "centroid_distance": round(closest_distance, 3)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


# =========================
# Run App
# =========================

if __name__ == "__main__":
    app.run(debug=True)