from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import logging
import os

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔐 Change this to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Paths
model_path = os.path.join(os.getcwd(), "app/model.pkl")
encoder_path = os.path.join(os.getcwd(), "app/encoders.pkl")

# Debug prints
print("Looking for model at:", model_path)
print("Looking for encoders at:", encoder_path)

try:
    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
except Exception as e:
    logging.error(f"❌ Failed to load model or encoders: {e}", exc_info=True)
    raise RuntimeError(f"Error loading model or encoders: {str(e)}")

# Validate encoder keys
required_keys = ['gender', 'mood', 'personality', 'mlb']
if not all(key in encoders for key in required_keys):
    raise HTTPException(status_code=500, detail="Missing necessary encoders. Check model files.")

label_encoder = {
    'gender': encoders['gender'],
    'mood': encoders['mood'],
    'personality': encoders['personality']
}
mlb = encoders['mlb']

# Print allowed classes
print("Allowed genders:", label_encoder['gender'].classes_)
print("Allowed personalities:", label_encoder['personality'].classes_)
print("Allowed moods:", label_encoder['mood'].classes_)

# Schemas
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

        features = np.array([[gender_encoded, mood_encoded, personality_encoded]])
        predictions = model.predict(features)

        predicted_activities = mlb.inverse_transform(predictions)[0]

        return PredictionResponse(
            recommended_activities=sorted(predicted_activities) if predicted_activities else ["No recommendations available"]
        )

    except ValueError as ve:
        logging.error(f"Input error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
