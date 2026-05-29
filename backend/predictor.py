import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load


# load .env content to env vars
load_dotenv()


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_FILENAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# load the trained model only once (module-level cache)
model = load(MODEL_PATH)
logging.info("Model loaded successfully.")

def predict(input_data: dict):

    text = input_data["content"]

    prediction = int(model.predict([text])[0])

    logging.info(f"Model provided a prediction: {prediction}")

    return {
        "prediction": prediction
    }


# # example usage
# sample_input = {
#     "content": "Congratulations! You've won a free ticket to the Bahamas. Click here to claim now!"
# }
# result = predict(input_data=sample_input)
# print(result)
