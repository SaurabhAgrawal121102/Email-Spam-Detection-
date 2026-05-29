# 📧 Email Spam Detection System

A complete End-to-End Machine Learning project for detecting spam emails using Natural Language Processing (NLP), Scikit-Learn, FastAPI, and Streamlit.

---

## 🚀 Live Demo

Access the deployed application here:

**http://32.198.105.17:8501**

Users can enter email content and instantly receive a prediction indicating whether the message is **Spam** or **Not Spam**.

---

## 📌 Project Overview

Email spam remains one of the most common cybersecurity and communication challenges. This project uses Machine Learning and Natural Language Processing techniques to classify emails as either:

* **Spam (1)**
* **Not Spam / Ham (0)**

The application consists of:

* Machine Learning Training Pipeline
* FastAPI Backend API
* Streamlit Frontend
* EC2 Cloud Deployment

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Scikit-Learn
* Pandas
* NumPy
* Joblib

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* Streamlit

### Deployment

* AWS EC2
* Ubuntu Linux

---

## 📂 Project Structure

```text
Email-Spam-Detection/
│
├── backend/
│   ├── main.py
│   └── predictor.py
│
├── frontend/
│   └── app.py
│
├── dataset/
│   └── spam.csv
│
├── model_dir/
│   └── model_spam.joblib
│
├── logs/
│
├── requirements.txt
├── README.md
└── .env
```

---

## 📊 Dataset

Dataset contains SMS/Email messages labeled as:

| Label | Description    |
| ----- | -------------- |
| 0     | Not Spam (Ham) |
| 1     | Spam           |

Target Column:

```python
v1
```

Text Column:

```python
v2
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

### Text Cleaning

* Convert text to lowercase
* Remove URLs
* Remove HTML tags
* Remove unwanted special characters
* Normalize whitespace

Example:

```python
clean_text(text)
```

---

## 🔤 Feature Engineering

TF-IDF Vectorization was used to convert text into numerical features.

Configuration:

```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)
```

---

## 🤖 Model Training

### Algorithm Used

```python
LinearSVC(
    C=1,
    class_weight="balanced"
)
```

### Training Pipeline

```python
Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("model", LinearSVC(...))
])
```

### Why LinearSVC?

* Fast training
* Excellent performance on text classification
* Handles high-dimensional sparse TF-IDF features effectively
* Lower memory usage than many ensemble methods

---

## 📈 Model Performance

### Accuracy

```text
98.74%
```

### Confusion Matrix

```text
[[964   2]
 [ 12 137]]
```

### Classification Report

```text
              precision    recall  f1-score   support

           0     0.9877    0.9979    0.9928       966
           1     0.9856    0.9195    0.9514       149

    accuracy                         0.9874      1115
   macro avg     0.9867    0.9587    0.9721      1115
weighted avg     0.9874    0.9874    0.9872      1115
```

---

## 💾 Model Saving

The trained pipeline is saved using Joblib:

```python
joblib.dump(model_pipeline, "model_spam.joblib")
```

---

## 🔮 Prediction Pipeline

The prediction workflow:

1. User enters email content.
2. Streamlit sends request to FastAPI.
3. FastAPI calls predictor module.
4. Trained model predicts spam/not spam.
5. Result returned to user.

Example Input:

```text
Congratulations! You have won a free iPhone.
Click here to claim your prize now.
```

Prediction:

```text
Spam
```

Example Input:

```text
Hi John,

Can we schedule a meeting tomorrow at 10 AM?

Regards,
Sarah
```

Prediction:

```text
Not Spam
```

---

## 🌐 FastAPI Endpoints

### Health Check

```http
GET /status
```

Response:

```json
{
    "status": "API is running"
}
```

### Prediction Endpoint

```http
POST /predict
```

Request:

```json
{
    "content": "Free gift waiting for you!"
}
```

Response:

```json
{
    "prediction": 1
}
```

---

## 🎨 Streamlit Frontend

The Streamlit application provides:

* Clean User Interface
* Email Content Input Box
* Real-Time Prediction
* Error Handling
* Backend API Integration

Run locally:

```bash
streamlit run frontend/app.py
```

---

## ☁️ AWS Deployment

Services deployed on:

* AWS EC2 (Ubuntu)
* FastAPI using Uvicorn
* Streamlit Frontend

Start FastAPI:

```bash
nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

Start Streamlit:

```bash
nohup streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
```

---

## 📦 Libraries Used

```text
scikit-learn
numpy
pandas
joblib
python-dotenv
pydantic
fastapi
uvicorn
requests
streamlit
xgboost
```

---

## 👨‍💻 Author

Saurabh Agrawal

Machine Learning | Data Science | Python | NLP | FastAPI | Streamlit | AWS

---

## ⭐ Future Improvements

* Probability score output
* Model monitoring
* Docker deployment
* CI/CD pipeline
* Email attachment scanning
* Advanced NLP preprocessing
* Transformer-based spam detection (BERT)

---
