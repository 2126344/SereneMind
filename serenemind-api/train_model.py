import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Paths
DATA_PATH = "cleaned_serenemind_data.csv"
OUTPUT_DIR = "app"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df[df['Gender'].isin(['Male', 'Female'])]
df = df[['Gender', 'Personality', 'Anxious', 'Depressed', 'Neutral', 'Stressed', 'Happy']].dropna()

# Long format
moods = ['Anxious', 'Depressed', 'Neutral', 'Stressed', 'Happy']
mood_dfs = []

for mood in moods:
    temp = df[['Gender', 'Personality', mood]].copy()
    temp.columns = ['Gender', 'Personality', 'Activities']
    temp['Mood'] = mood
    temp['Activities'] = temp['Activities'].apply(
        lambda x: [i.strip() for i in str(x).split(',')] if isinstance(x, str) else []
    )
    mood_dfs.append(temp)

long_df = pd.concat(mood_dfs, ignore_index=True)

# Encode input features
le_gender = LabelEncoder()
le_personality = LabelEncoder()
le_mood = LabelEncoder()

long_df['Gender_enc'] = le_gender.fit_transform(long_df['Gender'])
long_df['Personality_enc'] = le_personality.fit_transform(long_df['Personality'])
long_df['Mood_enc'] = le_mood.fit_transform(long_df['Mood'])

X = long_df[['Gender_enc', 'Mood_enc', 'Personality_enc']]
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(long_df['Activities'])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train model
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
model.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = model.predict(X_test)

# Calculate accuracy for each label
accuracies = []
for i in range(Y.shape[1]):
    acc = accuracy_score(Y_test[:, i], Y_pred[:, i])
    label = mlb.classes_[i]
    print(f"Accuracy for '{label}': {acc:.2f}")
    accuracies.append(acc)

print(f"\nAverage accuracy across all labels: {np.mean(accuracies):.2f}")

# Save model and encoders
joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
joblib.dump({
    'gender': le_gender,
    'mood': le_mood,
    'personality': le_personality,
    'mlb': mlb
}, os.path.join(OUTPUT_DIR, "encoders.pkl"))

print("Model training, evaluation complete, and saved.")

