from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)

model = joblib.load("app/model.pkl")
encoders = joblib.load("app/encoders.pkl")

label_encoder = {
    'gender': encoders['gender'],
    'mood': encoders['mood'],
    'personality': encoders['personality']
}
mlb = encoders['mlb']

class UserInput(BaseModel):
    gender: str
    personality: str
    mood: str

class PredictionResponse(BaseModel):
    recommended_activities: List[str]

@app.post("/predict", response_model=PredictionResponse)
def predict_activities(user: UserInput):
    try:
        gender_encoded = label_encoder['gender'].transform([user.gender])[0]
        personality_encoded = label_encoder['personality'].transform([user.personality])[0]
        mood_encoded = label_encoder['mood'].transform([user.mood])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    features = np.array([[gender_encoded, mood_encoded, personality_encoded]])
    predictions = model.predict(features)
    predicted_activities = mlb.inverse_transform(predictions)[0]

    return PredictionResponse(
        recommended_activities=sorted(predicted_activities) if predicted_activities else ["No recommendations available"]
    )
