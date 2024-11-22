from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    Month: int
    Sex: int
    MaritalStatus: int
    Age: int
    Fault: int
    VehicleCategory: int
    VehiclePrice: int
    RepNumber: int
    Deductible: int
    DriverRating: int
    PastNumberOfClaims: int
    AgeOfVehicle: int
    NumberOfSuppliments: int
    AddressChange_Claim: int
    NumberOfCars: int
    Year: int
    BasePolicy: int
    Make_Honda: int
    Make_Mazda: int
    Make_Pontiac: int
    Make_Toyota: int

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "Favicon not available."}

@app.post("/predict/")
async def predict(data: InputData):
    # Parse input data
    # Format the input data for model prediction
    input_array = np.array([[9,1,1,38,1,1,1,11,400,4,1,0,2,0,0,1996,2,0,0,1,0]]) # output should be 0
    
    # Make prediction
    prediction = model.predict(input_array)
    prediction_prob = model.predict_proba(input_array)
    
    # Return response
    return {
        "prediction": prediction.tolist(),
        "probabilities": prediction_prob.tolist()
    }