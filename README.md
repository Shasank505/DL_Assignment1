# Deep Learning Assignment 1

This repository contains the implementation of a deep learning model along with a FastAPI application for serving predictions. Below is an overview of the project structure, installation instructions, and functionality.

---

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd DL_Assignment1

   ```

2. Install the required dependencies:

````bash
pip install -r requirement.txt

## Project Structure
DL_Assignment1/
├── Deep Learning Assignment 1 Model Report.pdf  # Report detailing the model
├── [README.md](http://_vscodecontentref_/1)                                    # Project documentation
├── [requirement.txt](http://_vscodecontentref_/2)                              # Python dependencies
├── fastapi_app/                                 # FastAPI application
│   ├── encoder.pkl                              # Pre-trained encoder
│   ├── main.py                                  # FastAPI server implementation
│   ├── scaler.pkl                               # Pre-trained scaler
│   ├── [test_api.ipynb](http://_vscodecontentref_/3)                           # Notebook for testing the API
│   ├── trained_model.h5                         # Trained model file
│   ├── trained_model.weights.h5                 # Model weights
│   └── __pycache__/                             # Compiled Python files
├── model_training/                              # Model training and preprocessing
│   ├── __init__.py                              # Package initialization
│   ├── [dlmodel.ipynb](http://_vscodecontentref_/4)                            # Deep learning model training
│   ├── [preprocessed.ipynb](http://_vscodecontentref_/5)                       # Data preprocessing steps
│   └── __pycache__/                             # Compiled Python files

## Functionality

1. Model Training
The model_training/ directory contains Jupyter notebooks for training and preprocessing the data.
dlmodel.ipynb trains a deep learning model using TensorFlow.
preprocessed.ipynb handles data cleaning and feature engineering.

2. FastAPI Application
The fastapi_app/ directory contains a FastAPI application for serving predictions.
main.py implements the API endpoints.
test_api.ipynb provides an example of how to interact with the API.

3. Pre-trained Artifacts
encoder.pkl and scaler.pkl are used for preprocessing input data.
trained_model.h5 and trained_model.weights.h5 are the saved model and its weights.

## Usage

1. Start the FastAPI server:
```bash
    uvicorn fastapi_app.main:app --reload
````
