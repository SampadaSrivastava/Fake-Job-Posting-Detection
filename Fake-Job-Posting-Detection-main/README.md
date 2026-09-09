# JobGuard AI — Fake Job Posting Detection

## 1. Project Overview

JobGuard AI is a machine learning and Natural Language Processing (NLP) based system designed to identify potentially fraudulent job postings.

The system analyzes job-posting text and classifies it as either:

- Legitimate Job Posting
- Fraudulent Job Posting

A Flask web application provides a user-friendly interface for testing unseen job postings.

---

## 2. Objectives

- Analyze job-posting data using exploratory data analysis.
- Handle missing values and duplicate records.
- Combine important textual job-posting features.
- Apply NLP preprocessing.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple machine learning models.
- Evaluate models using Accuracy, Precision, Recall, and F1 Score.
- Select an appropriate final model.
- Deploy the final model using Flask.
- Provide a practical job-posting prediction interface.

---

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF
- LinearSVC
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Flask
- HTML
- CSS
- JavaScript
- Joblib
- VS Code

---

## 4. Dataset

The project uses a job-posting dataset containing textual and structured information about employment opportunities.

Important textual features used in the final model include:

- `title`
- `company_profile`
- `description`
- `requirements`
- `benefits`

These fields are combined into a single text representation before TF-IDF feature extraction.

---

## 5. Machine Learning Pipeline

The project follows this workflow:

Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Text Feature Engineering
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Train-Test Split
↓
Model Training
↓
Model Evaluation
↓
Model Comparison
↓
Final Model Selection
↓
Flask Deployment

---

## 6. TF-IDF Configuration

The final TF-IDF vectorizer uses:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

This allows the model to learn from both individual words and two-word combinations.

7. Machine Learning Models

The following models were evaluated:

Logistic Regression
Multinomial Naive Bayes
Random Forest
Linear Support Vector Machine (LinearSVC)
8. Model Comparison
Model	Accuracy	Precision	Recall	F1 Score
Linear SVM	97.96%	75.77%	84.97%	80.11%
Random Forest	98.13%	100.00%	61.27%	75.99%
Logistic Regression	96.36%	58.24%	87.86%	70.05%
Multinomial Naive Bayes	96.90%	86.90%	42.20%	56.81%
9. Final Model Performance

After selecting and retraining the final model, the test performance was:

Metric	Score
Accuracy	98.27%
Precision	79.37%
Recall	86.71%
F1 Score	82.87%
Classification Report
Class	Precision	Recall	F1 Score
Legitimate	0.99	0.99	0.99
Fraudulent	0.79	0.87	0.83

Training samples:

14304

Testing samples:

3576
10. Feature Combination Analysis

Different combinations of textual features were tested.

Feature Combination	Accuracy	Precision	Recall	F1 Score
Title + Description	97.18%	68.00%	78.61%	72.92%
Title + Requirements	95.55%	52.80%	76.30%	62.41%
Description + Requirements	97.04%	65.73%	80.92%	72.54%
Title + Description + Requirements	97.04%	66.03%	79.77%	72.25%
All Features	98.27%	79.37%	86.71%	82.87%

The combination of all selected textual fields provided the strongest overall performance.

11. Raw Text vs Cleaned Text
Preprocessing	Accuracy	Precision	Recall	F1 Score
Raw Text	98.27%	79.37%	86.71%	82.87%
Cleaned Text	98.24%	79.26%	86.13%	82.55%

The difference was very small, with raw text performing slightly better in the tested configuration.

12. Web Application

The final model is integrated into a Flask application called JobGuard AI.

The application allows users to:

Paste a job posting.
Submit the posting for analysis.
Convert the text using the saved TF-IDF vectorizer.
Generate a machine learning prediction.
View the risk level.
Receive a recommendation.

The application provides results such as:

Legitimate Job Posting
Risk Level: Low Risk

or

Fraudulent Job Posting
Risk Level: High Risk

The system also provides a disclaimer that predictions should not be considered a definitive determination of whether a job is fraudulent.

13. Project Structure
Fake-Job-Posting-Detection/
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── model.pkl
│   └── tfidf.pkl
│
├── notebooks/
│   └── Fake_Job_Posting_Detection.ipynb
│
├── src/
│   └── predict.py
│
├── requirements.txt
│
└── README.md
14. Running the Project
Step 1 — Create/activate virtual environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1
Step 2 — Install dependencies
pip install -r requirements.txt
Step 3 — Run Flask application

From the project root:

python app\app.py
Step 4 — Open the application

Open the Flask address shown in the terminal, normally:

http://127.0.0.1:5000
15. Limitations
The model depends on the quality and representativeness of the training dataset.
New fraud patterns may not be recognized immediately.
Text-based classification cannot verify an employer independently.
False positives and false negatives are possible.
Dataset imbalance can affect evaluation.
A prediction should not be treated as definitive proof that a job is fraudulent.
16. Future Improvements

Future versions could include:

Larger and more recent datasets.
Transformer-based NLP models such as BERT.
Company and domain verification.
URL and email analysis.
Salary anomaly detection.
Explainable AI.
Probability-based risk scoring.
Continuous model retraining.
Real-time job-posting analysis.
17. Conclusion

JobGuard AI demonstrates the application of machine learning and NLP to the problem of fraudulent job-posting detection.

The project covers the complete machine learning workflow, including data analysis, preprocessing, feature engineering, TF-IDF vectorization, model training, evaluation, comparison, and deployment.

The final system achieved:

98.27% Accuracy

79.37% Precision

86.71% Recall

82.87% F1 Score

The trained model was successfully integrated into a Flask web application, providing a practical interface for analyzing unseen job postings.

18. Disclaimer

JobGuard AI provides a machine-learning-based prediction and should not be considered a definitive determination of whether a job posting is fraudulent.

Users should independently verify the employer, company information, job details, contact information, and any payment-related requests before applying or sharing personal information.