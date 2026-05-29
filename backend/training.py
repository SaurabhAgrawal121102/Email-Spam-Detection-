import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, data
import joblib
import os
from dotenv import load_dotenv
import logging
import re 
from sklearn.linear_model import LogisticRegression
load_dotenv()
def clean_text(s:str)->str:
    """
    Simple text cleaning:
    - lowercasing
    - remove URLs
    - remove HTML tags
    - keep letters/numbers and basic punctuation
    - normalize spaces
    """
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)          # URLs
    s = re.sub(r"<.*?>", " ", s)                    # HTML tags
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\']", " ", s)   # keep basic chars
    s = re.sub(r"\s+", " ", s).strip()               # normalize spaces
    return s
def train_model():

    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    DATASET_NAME = os.getenv("DATASET_NAME")
    MODEL_DIR = os.getenv("MODEL_DIR")
    LOG_NAME = os.getenv("LOG_NAME")
    TARGET_COLUMN = os.getenv("TARGET_COLUMN")
    TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
    LOG_DIR = os.getenv("LOG_DIR")

    if not all([PROJECT_ROOT, DATASET_NAME, MODEL_DIR, LOG_NAME, TARGET_COLUMN, LOG_DIR]):
        raise ValueError("Missing environment variables")

    logging.basicConfig(
        filename=os.path.join(LOG_DIR, LOG_NAME),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

    try:
        dataset_path = os.path.join(PROJECT_ROOT, DATASET_NAME)
        df = pd.read_csv(dataset_path, encoding='latin-1')
        logging.info("Dataset loaded successfully.")

        df["v1"] = df["v1"].replace({
           "ham":"not spam",
            "spam":"spam"
        })
        df["v1"] = df["v1"].map({"spam":1,"not spam":0})
        df["content"] = df["v2"].apply(clean_text)
        X = df["content"]
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )
        logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        model_pipeline = Pipeline(
        steps = [
        ("tfidf",TfidfVectorizer(
            max_features = 50000,
            ngram_range = (1,2),
            stop_words = "english",
            min_df =2
        )),
        ("model",LogisticRegression(class_weight = "balanced"))
    ]
)
        model_pipeline.fit(X_train,y_train)
        y_pred_train = model_pipeline.predict(X_train)
        y_proba_train = model_pipeline.predict_proba(X_train)[:, 1]

        acc_train = accuracy_score(y_train, y_pred_train)
        cm_train = confusion_matrix(y_train, y_pred_train)

        logging.info("\n--- Train Metrics ---")
        logging.info(f"Accuracy : {acc_train:.4f}")

        logging.info("\nConfusion Matrix [ [TN FP], [FN TP] ]:")
        logging.info(cm_train)

        logging.info("\nClassification Report:")
        logging.info(classification_report(y_train, y_pred_train, digits=4))

        
       
        logging.info("Model trained successfully.")

    except Exception as e:
        logging.error(f"Training failed: {e}")
if __name__ == "__main__":
    train_model()