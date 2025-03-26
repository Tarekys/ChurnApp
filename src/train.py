# Training Deep Learning Model using PyTorch

# Data Split
X = data.drop(columns=["Churn"])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
#----------------------------------------------------------------------
# Hyperparameters
num_epochs = 100
batch_size = 64
learning_rate = 0.005
patience = 3
lr_factor = 0.5
weight_decay  = 1e-4

# StandardScaler
import joblib

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')


# Convert y_train and y_test to NumPy arrays before creating tensors
y_train = y_train.values  # Extract NumPy array from y_train Series
y_test = y_test.values    # Extract NumPy array from y_test Series


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# Create a DataLoader to improve data processing during training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#----------------------------------------------------------------------
## Neural Network Architecture
class ChurnModel(nn.Module):
  def __init__(self, input_dim):

    super(ChurnModel, self).__init__()

    self.layers = nn.Sequential(

        nn.Linear(input_dim, 256), # The first layer
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(256, 128), # The second layer
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, 64), # 3th layer
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(64, 32), # 4th layer
        nn.ReLU(),

        nn.Linear(32,1), # output layer
        nn.Sigmoid()

    )

  # Define forward pass
  def forward(self,x):
    return self.layers(x)

# Define the Model
input_dim = X_train.shape[1]
model = ChurnModel(input_dim)


# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Using the Adam optimizer

# Learning rate scheduler (optional but recommended)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=patience, factor=lr_factor, verbose=True
)
#-----------------------------------------------------------------------
## Training the Model
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for epoch in range(num_epochs):
        model.train()  # Training Mode
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Remove squeezing from the output
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Remove float() as it's already float32
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

#----------------------------------------------------------------------
        # Validation      
        def evaluate_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Remove squeezing from the output
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Remove float() as it's already float32
            total_loss += loss.item()

            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
#----------------------------------------------------------------------
## Full Training process
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)
evaluate_model(model, test_loader, criterion)