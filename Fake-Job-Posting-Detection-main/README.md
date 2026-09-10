# 🚨 FraudHire – Fake Job Posting Detection

> A Machine Learning and Natural Language Processing (NLP) based system that detects potentially fraudulent job postings using job-description text and classification models.

## 📌 Overview

**FraudHire** is a Machine Learning and NLP project developed to identify potentially fraudulent job advertisements.

Online job platforms contain a large number of employment opportunities, but some postings may be misleading or fraudulent. FraudHire analyzes the textual information in a job posting and predicts whether it is:

* ✅ **Legitimate**
* 🚨 **Fraudulent**

The project implements an end-to-end machine learning workflow, starting from data analysis and cleaning, followed by text feature engineering, TF-IDF vectorization, model training and evaluation, and finally deployment through a Flask web application.

---

## 🎯 Objectives

The main objectives of FraudHire are to:

* Analyze a dataset of job postings.
* Perform exploratory data analysis.
* Handle missing values and duplicate records.
* Identify useful textual features.
* Combine multiple job-posting fields for classification.
* Apply NLP and TF-IDF feature extraction.
* Train and compare different machine learning algorithms.
* Evaluate models using multiple classification metrics.
* Select and save the final model.
* Integrate the model into a Flask web application.
* Provide a simple interface for analyzing new job postings.

---

## 🛠️ Technologies Used

### Programming & Data Processing

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest
* LinearSVC

### NLP

* TF-IDF Vectorization
* Text preprocessing
* Unigram and Bigram features

### Visualization

* Matplotlib
* Seaborn

### Web Development

* Flask
* HTML
* CSS
* JavaScript

### Other Tools

* Joblib
* VS Code

---

## 📊 Dataset

The project uses a dataset containing job-posting information, including both textual and structured attributes.

The following textual fields are used by the final model:

```text
title
company_profile
description
requirements
benefits
```

These fields are combined to create a single text input for the NLP classification pipeline.

---

## 🔄 Project Workflow

```text
                  Job Posting Dataset
                         │
                         ▼
                  Data Cleaning
                         │
                         ▼
                 Exploratory Analysis
                         │
                         ▼
              Text Feature Engineering
                         │
                         ▼
                NLP / Text Processing
                         │
                         ▼
                  TF-IDF Vectorization
                         │
                         ▼
                   Train-Test Split
                         │
                         ▼
                  Model Training
                         │
                         ▼
                  Model Evaluation
                         │
                         ▼
                 Model Comparison
                         │
                         ▼
                 Final Model Selection
                         │
                         ▼
                  Flask Deployment
                         │
                         ▼
                  FraudHire Web App
```

---

## 🔤 TF-IDF Feature Extraction

FraudHire uses TF-IDF to transform job-posting text into numerical features that can be processed by machine learning algorithms.

The final vectorizer configuration is:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

### Why TF-IDF?

TF-IDF assigns importance to words based on how frequently they occur in a document compared with the complete dataset.

The `(1, 2)` n-gram configuration allows the model to consider both:

* **Unigrams:** individual words
* **Bigrams:** two-word combinations

This helps capture meaningful phrases within job postings.

---

## 🤖 Models Evaluated

Four classification algorithms were trained and compared:

1. **Logistic Regression**
2. **Multinomial Naive Bayes**
3. **Random Forest**
4. **Linear Support Vector Machine (LinearSVC)**

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📈 Model Comparison

| Model                   | Accuracy | Precision | Recall | F1 Score |
| ----------------------- | -------: | --------: | -----: | -------: |
| Linear SVM              |   97.96% |    75.77% | 84.97% |   80.11% |
| Random Forest           |   98.13% |   100.00% | 61.27% |   75.99% |
| Logistic Regression     |   96.36% |    58.24% | 87.86% |   70.05% |
| Multinomial Naive Bayes |   96.90% |    86.90% | 42.20% |   56.81% |

Model selection was based on overall performance rather than accuracy alone, with particular attention to the ability to identify fraudulent postings.

---

## 🏆 Final Model Performance

After selecting and retraining the final model, FraudHire achieved the following performance on the test set:

| Metric        |      Score |
| ------------- | ---------: |
| **Accuracy**  | **98.27%** |
| **Precision** | **79.37%** |
| **Recall**    | **86.71%** |
| **F1 Score**  | **82.87%** |

### Classification Report

| Class      | Precision | Recall | F1 Score |
| ---------- | --------: | -----: | -------: |
| Legitimate |      0.99 |   0.99 |     0.99 |
| Fraudulent |      0.79 |   0.87 |     0.83 |

### Dataset Split

```text
Training Samples : 14,304
Testing Samples  : 3,576
```

The final model provides a strong balance between precision and recall for the fraudulent class.

---

## 🔍 Feature Combination Experiment

Different combinations of textual fields were tested to determine which representation provided the best results.

| Feature Combination                |   Accuracy |  Precision |     Recall |   F1 Score |
| ---------------------------------- | ---------: | ---------: | ---------: | ---------: |
| Title + Description                |     97.18% |     68.00% |     78.61% |     72.92% |
| Title + Requirements               |     95.55% |     52.80% |     76.30% |     62.41% |
| Description + Requirements         |     97.04% |     65.73% |     80.92% |     72.54% |
| Title + Description + Requirements |     97.04% |     66.03% |     79.77% |     72.25% |
| **All Features**                   | **98.27%** | **79.37%** | **86.71%** | **82.87%** |

The combination of all selected textual fields produced the strongest overall performance.

### Final Text Input

```text
Title
+
Company Profile
+
Description
+
Requirements
+
Benefits
```

---

## 🧪 Raw Text vs Cleaned Text

An additional experiment compared raw text with cleaned text.

| Approach     |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------ | ---------: | ---------: | ---------: | ---------: |
| **Raw Text** | **98.27%** | **79.37%** | **86.71%** | **82.87%** |
| Cleaned Text |     98.24% |     79.26% |     86.13% |     82.55% |

The performance difference was minimal. In the tested configuration, raw text achieved slightly better results.

---

## 🌐 FraudHire Web Application

The trained machine learning model is integrated into a Flask web application.

The application allows a user to enter or paste a job posting and receive a prediction.

### Application Flow

```text
User enters job posting
          ↓
Text is received by Flask
          ↓
Saved TF-IDF vectorizer transforms the text
          ↓
Trained ML model generates prediction
          ↓
Prediction is displayed
          ↓
Risk level and recommendation are shown
```

### Example Output

**Legitimate Posting**

```text
Prediction: Legitimate Job Posting
Risk Level: Low Risk
```

**Fraudulent Posting**

```text
Prediction: Fraudulent Job Posting
Risk Level: High Risk
```

The application also displays a disclaimer explaining that the prediction is not definitive proof of fraud.

---

## 📁 Project Structure

```text
FraudHire/
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
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/SampadaSrivastava/Fake-Job-Posting-Detection.git
cd Fake-Job-Posting-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Flask Application

From the project root:

```bash
python app\app.py
```

### 6. Open the Application

Visit:

```text
http://127.0.0.1:5000
```

---

## 💡 What I Learned

Working on FraudHire helped me gain practical experience in:

* Data cleaning and exploratory analysis
* NLP-based text classification
* Text feature engineering
* TF-IDF vectorization
* Machine learning model training
* Model comparison
* Classification metrics
* Handling imbalanced classification
* Saving models using Joblib
* Flask application development
* Integrating ML models with a web interface

---

## ⚠️ Limitations

FraudHire has several limitations:

* Performance depends on the quality and representativeness of the dataset.
* Previously unseen fraud patterns may not be detected.
* Text classification alone cannot verify an employer.
* False positives and false negatives are possible.
* Dataset imbalance can affect model performance.
* A model prediction should not be treated as definitive proof of fraud.

---

## 🔮 Future Improvements

Possible improvements for future versions include:

* Use larger and more recent datasets.
* Experiment with Transformer models such as BERT.
* Add company and domain verification.
* Analyze URLs included in job postings.
* Detect suspicious email addresses.
* Add salary anomaly detection.
* Implement explainable AI.
* Generate probability-based risk scores.
* Support real-time job-posting analysis.
* Continuously retrain the model with new fraud patterns.
* Develop a browser extension for detecting suspicious job listings.

---

## 🎯 Conclusion

**FraudHire** demonstrates how Machine Learning and NLP can be applied to the practical problem of detecting potentially fraudulent job postings.

The project implements a complete workflow covering:

**Data Analysis → Feature Engineering → NLP → TF-IDF → Machine Learning → Evaluation → Flask Deployment**

The final system achieved:

* **98.27% Accuracy**
* **79.37% Precision**
* **86.71% Recall**
* **82.87% F1 Score**

The trained model was successfully integrated into a Flask application, allowing users to analyze previously unseen job-posting text through a simple web interface.

---

## 👩‍💻 Author

### Sampada Srivastava

B.Tech Computer Science Engineering | Machine Learning & Data Analytics Enthusiast

