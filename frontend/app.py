import os
import requests
from dotenv import load_dotenv
import streamlit as st

# load environment variables
load_dotenv()

API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

st.title("Email Spam Detection")

st.write("Enter the content of the email to check if it's spam or not.")

email_content = st.text_area("Email Content", height=200)

if st.button("Predict"):

    if not email_content.strip():
        st.warning("Please enter the email content.")

    else:
        with st.spinner("Predicting..."):

            try:
                response = requests.post(
                    f"{API_URL}",
                    json={"content": email_content}
                )

                response.raise_for_status()

                result = response.json()

                prediction = result.get("prediction")

                # if API returns nested dict
                if isinstance(prediction, dict):
                    prediction = prediction.get("prediction")

                if prediction == 1:
                    st.error("The email is predicted to be SPAM.")
                else:
                    st.success("The email is predicted to be NOT SPAM.")

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the API: {e}")

            except ValueError:
                st.error("Invalid response from the API.")