import streamlit as st
import pandas as pd
import requests
import io

# Set page configuration
st.set_page_config(
    page_title="Car Accident Fraud Detection",
    page_icon="🚗",
    layout="wide",
)

# Title and description
st.title("🚗 Car Accident Fraud Detection")
st.write("""
Welcome to the Car Accident Fraud Detection app.
Fill in the insurance claim details manually or upload a CSV file to predict if a claim is fraudulent.
""")

# Option to choose between manual input and file upload
option = st.radio("Choose input method:", ('Manual Input', 'Upload CSV File'))

if option == 'Manual Input':
    st.header("Manual Input")
    
    # Create form for user input
    with st.form("input_form"):
        st.write("## Insurance Claim Details")
        # Organize inputs into columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Month = st.selectbox('Month', [str(i) for i in range(1, 13)])
            Make = st.selectbox('Make', ['Honda', 'Ford', 'Toyota', 'Mazda', 'Chevrolet', 'Pontiac'])
            Sex = st.selectbox('Sex', ['Male', 'Female'])
            MaritalStatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
            Age = st.number_input('Age', min_value=16, max_value=100, value=30)
            Fault = st.selectbox('Fault', ['Policy Holder', 'Third Party'])
            VehicleCategory = st.selectbox('Vehicle Category', ['Sedan', 'Sports Car', 'SUV'])
        
        with col2:
            VehiclePrice = st.selectbox('Vehicle Price', ['Less than 20,000', '20,000 to 40,000', 'More than 40,000'])
            RepNumber = st.number_input('Rep Number', min_value=0, max_value=20, value=1)
            Deductible = st.number_input('Deductible', min_value=0, max_value=1000, value=500)
            DriverRating = st.number_input('Driver Rating', min_value=0, max_value=5, value=3)
            PastNumberOfClaims = st.selectbox('Past Number of Claims', ['None', '1', '2 to 4', 'More than 4'])
            AgeOfVehicle = st.selectbox('Age of Vehicle', ['New', '2 Years', '3 Years', '4 Years', '5 Years', 'More than 5 Years'])
            WitnessPresent = st.selectbox('Witness Present', ['Yes', 'No'])
        
        with col3:
            AgentType = st.selectbox('Agent Type', ['External', 'Internal'])
            NumberOfSuppliments = st.selectbox('Number of Supplements', ['None', '1 to 2', '3 to 5', 'More than 5'])
            AddressChange_Claim = st.selectbox('Address Change since Policy Inception', ['Under 6 Months', '1 Year', '2 to 3 Years', 'Over 4 Years'])
            NumberOfCars = st.selectbox('Number of Cars', ['1 Vehicle', '2 Vehicles', '3 to 4', '5 to 8', 'More than 8'])
            Year = st.number_input('Year', min_value=1990, max_value=2023, value=2023)
            BasePolicy = st.selectbox('Base Policy', ['Liability', 'Collision', 'All Perils'])
        
        # Submit button
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare data
        input_data = {
            'Month': [Month],
            'Make': [Make],
            'Sex': [Sex],
            'MaritalStatus': [MaritalStatus],
            'Age': [Age],
            'Fault': [Fault],
            'VehicleCategory': [VehicleCategory],
            'VehiclePrice': [VehiclePrice],
            'RepNumber': [RepNumber],
            'Deductible': [Deductible],
            'DriverRating': [DriverRating],
            'PastNumberOfClaims': [PastNumberOfClaims],
            'AgeOfVehicle': [AgeOfVehicle],
            'WitnessPresent': [WitnessPresent],
            'AgentType': [AgentType],
            'NumberOfSuppliments': [NumberOfSuppliments],
            'AddressChange_Claim': [AddressChange_Claim],
            'NumberOfCars': [NumberOfCars],
            'Year': [Year],
            'BasePolicy': [BasePolicy]
        }

        df = pd.DataFrame(input_data)

        # Convert DataFrame to CSV buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Send data to FastAPI for prediction
        try:
            # Replace with your FastAPI endpoint URL
            api_url = "http://127.0.0.1:8000/predict"

            files = {'file': ('input_data.csv', csv_buffer.getvalue(), 'text/csv')}
            response = requests.post(api_url, files=files)
            if response.status_code == 200:
                prediction = response.json()['predictions'][0]
                if prediction == 1:
                    st.error("🚩 **The claim is predicted to be **_fraudulent_**.**")
                else:
                    st.success("✅ **The claim is predicted to be legitimate.**")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

        if st.button("Predict"):
            # Send data to FastAPI for prediction
            try:
                # Replace with your FastAPI endpoint URL
                api_url = "http://127.0.0.1:8000/predict"

                files = {'file': ('uploaded_file.csv', uploaded_file.getvalue(), 'text/csv')}
                response = requests.post(api_url, files=files)
                if response.status_code == 200:
                    predictions = response.json()['predictions']
                    # Map numerical predictions to labels
                    prediction_labels = ['Not Fraud' if pred == 0 else 'Fraud' for pred in predictions]
                    st.write("### Prediction Results")
                    # Display the prediction labels
                    st.write(prediction_labels)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")