# Predictions Function--Eva
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
#----------------------------------------------------------------------
# Neural Architecture
#----------------------------------------------------------------------
class ChurnModel(nn.Module):
  def __init__(self, input_dim):

    super(ChurnModel, self).__init__()

    self.layers = nn.Sequential(
        nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
         # The second layer
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        # 3th layer
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        # 4th layer and output layer
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid()
    )
  def forward(self,x):
    return self.layers(x)
#----------------------------------------------------------------------
# Predictions Function
#----------------------------------------------------------------------
def predict_from_csv(csv_path, model_path, scaler_path):
    df = pd.read_csv(csv_path)

    # input features
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    X_new = df[features]

    # Download StandardScaler
    scaler = joblib.load(scaler_path)
    X_new = scaler.transform(X_new.to_numpy())  

    # Convert to PyTorch tensor
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

    # Load the model
    input_dim = X_new.shape[1] 
    model = ChurnModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Predictions
    with torch.no_grad():
        predictions = model(X_new_tensor)
        predicted_labels = (predictions > 0.5).float() 


    df['Churn Prediction'] = pd.Series(predicted_labels.numpy().astype(int).squeeze()).map({0: "Stay", 1: "Will Churn"})

    
    return df[['Churn Prediction']]

csv_file = r'F:\AI\VSC\Project1\src\df2.csv'
model_file = r'F:\AI\VSC\Project1\models\churn_model.pth'
scaler_file = r'F:\AI\VSC\Project1\models\scaler.pkl'

predictions = predict_from_csv(csv_file, model_file, scaler_file)
print(predictions)