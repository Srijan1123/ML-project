from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title = "customer chrun prediction API")

model = joblib.load("chrun_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
 return{"message":"cutomer chrun predictions API is running"}

@app.post("/predict")
def predict_chrun(features:dict):
 input_data = np.array([list(features.values())])
 input_scaled = scaler.transform(input_data)
 prediction = model.predict(input_scaled)[0]
 probability = model.predict_proba(input_scaled)[0][1]
 
 return{
  "chrun_prediction": int(prediction),
  "chrun_probability": float(probability)
 }
 
 
 