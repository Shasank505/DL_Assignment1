from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os

# Load model and scaler
# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained Keras model from the file
model = load_model(os.path.join(BASE_DIR, "trained_model.h5"))

# Load the scaler object (used for feature scaling during training)
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Define input schema with default demo numbers
class PredictionInput(BaseModel):
    # The input should be a list of 16 numerical features
    # Default demo numbers are provided for testing purposes
    features: list[float] = [2025, 3, 2, 2025, 3, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Endpoint to predict sales price based on input features.
    """
    try:
        # Step 1: Validate input length
        # Ensure that the input contains exactly 16 features
        if len(data.features) != 16:
            # Return an error if the input length is invalid
            return {"error": "Invalid input", "message": f"Expected 16 features, but got {len(data.features)}"}
        
        # Step 2: Prepare and scale input data
        try:
            # Convert the input list to a NumPy array and reshape it for the model
            input_data = np.array(data.features).reshape(1, -1)
            
            # Scale the input data using the preloaded scaler
            input_data = scaler.transform(input_data)
        except Exception as e:
            # Handle errors during data preprocessing
            return {"error": "Data preprocessing error", "message": str(e)}
        
        # Step 3: Make prediction
        try:
            # Use the trained model to make a prediction
            prediction = model.predict(input_data).flatten()[0]
            
            # Convert the prediction from NumPy float32 to Python float for better compatibility
            prediction = float(prediction)
        except Exception as e:
            # Handle errors during model prediction
            return {"error": "Model prediction error", "message": str(e)}
        
        # Step 4: Return the prediction
        # Return the predicted value as a JSON response
        return {"prediction": prediction}
    
    except ValidationError as ve:
        # Handle validation errors (e.g., invalid input types)
        return {"error": "Validation error", "message": ve.errors()}
    except Exception as e:
        # Handle any unexpected errors
        return {"error": "Unexpected error", "message": str(e)}

@app.get("/")
def home():
    """
    Home endpoint to check if the API is running.
    """
    return {"message": "API is running!"}