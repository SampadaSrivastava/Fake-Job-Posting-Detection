import os
import joblib


# Get the project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Paths to saved model and TF-IDF vectorizer
model_path = os.path.join(
    BASE_DIR,
    "models",
    "model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "models",
    "tfidf.pkl"
)


# Load final trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


def predict_job(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    if prediction == 0:

        return {
            "label": "Legitimate Job Posting",
            "type": "legitimate",
            "message": (
                "The model did not identify strong fraudulent "
                "patterns in this job posting."
            ),
            "recommendation": (
                "Always verify the employer, job details and "
                "application process before sharing personal information."
            )
        }

    return {
        "label": "Fraudulent Job Posting",
        "type": "fraudulent",
        "message": (
            "The model identified patterns that may be associated "
            "with fraudulent job advertisements."
        ),
        "recommendation": (
            "Avoid sending money or sensitive personal information "
            "until the employer and job opportunity have been verified."
        )
    }


if __name__ == "__main__":

    sample_job = """
Frontend Developer

We are seeking a Frontend Developer with experience in HTML, CSS, JavaScript and React.

Responsibilities include developing responsive web applications, fixing UI issues and collaborating with backend developers.

Candidates should have good communication and problem-solving skills.

Salary will be based on experience.
    """

    result = predict_job(sample_job)

    print("Prediction:", result)