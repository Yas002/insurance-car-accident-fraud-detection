import streamlit as st
import pandas as pd
import requests

# Set page configuration
st.set_page_config(
    page_title="🚗 Car Accident Fraud Detection",
    page_icon="🚗",
    layout="wide",
)

# Title and description
st.title("🚗 Car Accident Fraud Detection")
st.write("""
Welcome to the Car Accident Fraud Detection app.
Upload a CSV file containing insurance claim details to predict if a claim is fraudulent.
""")

st.header("Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())
        
        if st.button("Predict"):
            # Send data to FastAPI for prediction
            try:
                # Define your FastAPI endpoint URL
                backend_url = "http://backend:8000/predict"
                
                # Prepare the files dictionary for the POST request
                files = {'file': ('input_data.csv', uploaded_file.getvalue(), 'text/csv')}
                
                # Send POST request to the backend
                response = requests.post(backend_url, files=files)
                
                if response.status_code == 200:
                    predictions = response.json()['predictions']
                    
                    # Map numerical predictions to labels
                    prediction_labels = ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions]
                    
                    st.write("### Prediction Results")
                    # Display the prediction labels
                    st.write(prediction_labels)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"An error occurred while connecting to the backend: {e}")
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")
else:
    st.info("Please upload a CSV file to get started.")