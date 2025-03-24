from fastapi import FastAPI
import torch
import pickle
import numpy as np
from pydantic import BaseModel

# Import the model architecture
from model_training.copy_of_untitled7 import SalesPredictionNN  # Ensure the path is correct

# Initialize the model architecture
input_size = 10  # Replace with the actual input size used during training
model = SalesPredictionNN(input_size)

# Load the model weights
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Load preprocessing artifacts
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Define request body
class PredictionInput(BaseModel):
    features: list

app = FastAPI()

@app.post("/predict")
async def predict(data: PredictionInput):
    # Preprocess input
    input_data = np.array(data.features).reshape(1, -1)
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return {"prediction": prediction}

@app.get("/")
def home():
    return {"message": "Sales Prediction API is running!"}