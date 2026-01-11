from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model


app = FastAPI(title="Customer Purchase Prediction API")


model = load_model("customer_purchase_nn_model.h5")
scaler = joblib.load("customer_purchase_scaler.pkl")

class CustomerData(BaseModel):
    age: float
    annual_income: float
    time_on_site: float
    avg_session_time: float
    pages_visited: float
    previous_purchases: float
    gender: int
    device: int
    location: int
    discount_used: int


@app.post("/predict_purchase")
def predict_purchase(data: CustomerData):
 
    input_data = np.array([[
        data.age,
        data.annual_income,
        data.time_on_site,
        data.avg_session_time,
        data.pages_visited,
        data.previous_purchases,
        data.gender,
        data.device,
        data.location,
        data.discount_used
    ]])

    
    engagement_score = input_data[0][4] * input_data[0][2]  
    loyalty_score = input_data[0][5] * input_data[0][3]     

    input_data = np.hstack([input_data, [[engagement_score, loyalty_score]]])

    input_scaled = scaler.transform(input_data)

    
    pred = model.predict(input_scaled)
    pred_label = int((pred > 0.5).astype(int)[0][0])

    return {"prediction": pred_label, "probability": float(pred[0][0])}


@app.get("/")
def home():
    return {"message": "Customer Purchase Prediction API is running"}
