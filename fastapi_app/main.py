from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import os

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load preprocessing artifacts
with open(os.path.join(base_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(base_dir, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

# Load the trained Keras model
model = load_model(os.path.join(base_dir, "mobile_price_model.h5"))

# Define request body
class PredictionInput(BaseModel):
    features: list

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(data: PredictionInput):
    # Preprocess input
    input_data = np.array(data.features).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Scale the input features
    
    # Make prediction
    prediction = model.predict(input_data).flatten()[0]  # Get the first prediction
    
    return {"prediction": prediction}

@app.get("/")
def home():
    return {"message": "Sales Prediction API is running!"}