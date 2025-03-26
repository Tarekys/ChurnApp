from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import shutil
import pandas as pd
import torch
import torch.nn as nn
import joblib
import io
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#----------------------------------------------------------------------
# Neural Architecture
#----------------------------------------------------------------------
class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
# download the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/churn_model.pth")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models/scaler.pkl")

scaler = joblib.load(SCALER_PATH)

def load_model():
    input_dim = 17
    model = ChurnModel(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

#----------------------------------------------------------------------
# API Endpoint to Predict from CSV
#----------------------------------------------------------------------
@app.post("/predictions")

async def predict_from_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))

        features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]

        X_new = df[features].to_numpy()
        X_new = scaler.transform(X_new)
##F:\AI\VSC\Project1\main.py : uvicorn Project1.main:app --reload
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

        # predict
        with torch.no_grad():
            predictions = model(X_new_tensor)
            predicted_labels = (predictions > 0.5).float()


        df['Churn Prediction'] = pd.Series(predicted_labels.numpy().astype(int).squeeze()).map({0: "Stay", 1: "Will Churn"})
        return df[['Churn Prediction']].to_dict()

    except Exception as e:
        return {"error": str(e)}
