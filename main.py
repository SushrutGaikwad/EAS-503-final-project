from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model
model_path = "best_logistic_regression_model_without_yeo_johnson.pkl"
model = joblib.load(model_path)

# Define the request schema
class PredictionRequest(BaseModel):
    features: dict  # Input data as a dictionary of feature names and values

# Define the response schema
class PredictionResponse(BaseModel):
    prediction: str  # Predicted class as a readable string (<=50K or >50K)

# Create the FastAPI app
app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Convert input features to DataFrame
    input_df = pd.DataFrame([request.features])
    
    # Generate predictions
    prediction = model.predict(input_df)[0]
    
    # Map prediction to human-readable output
    readable_prediction = "<=50K" if prediction == 0 else ">50K"
    
    # Return the response
    return PredictionResponse(prediction=readable_prediction)
