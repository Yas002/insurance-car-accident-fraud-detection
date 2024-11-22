from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI()