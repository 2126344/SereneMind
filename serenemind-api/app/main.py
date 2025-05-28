from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import logging
import os

app = FastAPI()

# ✅ Enable CORS for external requests (Supabase, Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allows all origins (Change this to specific domains for security)
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Allows all headers
)

logging.basicConfig(level=logging.INFO)

# ✅ Correct model & encoder paths
model_path = os.path.join(os.getcwd(), "app/model.pkl")
encoder_path = os.path.join(os.getcwd(), "app/encoders.pkl")

try:
    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
except Exception as e:
    raise RuntimeError(f"Error loading model or encoders: {str(e)}")

# ✅ Ensure encoders exist before use
if 'gender' not in encoders or 'mood' not in encoders or 'personality' not in encoders or 'mlb' not in encoders:
    raise HTTPException(status_code=500, detail="Missing necessary encoders. Check model files.")

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

    predicted_activities = mlb.inverse_transform(predictions)[0] if predictions else []

    return PredictionResponse(
        recommended_activities=sorted(predicted_activities) if predicted_activities else ["No recommendations available"]
    )