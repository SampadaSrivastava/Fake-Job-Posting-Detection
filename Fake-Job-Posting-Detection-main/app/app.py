from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "tfidf.pkl"
)


# ==========================================
# LOAD FINAL MODEL
# ==========================================

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_job(text):

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_vector)[0]

    # SVM decision score
    decision_score = model.decision_function(text_vector)[0]

    if prediction == 1:

        result = {
            "type": "fraudulent",
            "label": "Fraudulent Job Posting",
            "message": (
                "The model detected patterns that may be "
                "associated with fraudulent job advertisements."
            ),
            "recommendation": (
                "Do not pay registration or security fees. "
                "Verify the company and employer before sharing "
                "personal or financial information."
            )
        }

        risk_level = "High Risk"

    else:

        result = {
            "type": "legitimate",
            "label": "Legitimate Job Posting",
            "message": (
                "The model did not identify strong fraudulent "
                "patterns in this job posting."
            ),
            "recommendation": (
                "Still verify the employer, job details and "
                "company information before applying."
            )
        }

        risk_level = "Low Risk"

    return result, risk_level, decision_score


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    risk_level = ""
    decision_score = None

    if request.method == "POST":

        job_description = request.form["job_description"]

        prediction, risk_level, decision_score = predict_job(
            job_description
        )

    return render_template(
        "index.html",
        prediction=prediction,
        risk_level=risk_level,
        decision_score=decision_score
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)