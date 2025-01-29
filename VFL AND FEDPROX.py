import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data_path = 'C:/Users/keerthan/Desktop/fed/gym_membership.csv' 
data = pd.read_csv(data_path)

# Check the columns in the dataset
print("Columns in the dataset:", data.columns.tolist())

# Preprocess the dataset: Convert date columns to datetime and handle missing values
for col in data.columns:
    if pd.api.types.is_string_dtype(data[col]):  # Check for string columns
        try:
            data[col] = pd.to_datetime(data[col])  # Convert to datetime if possible
        except ValueError:
            continue  # If conversion fails, leave it as is

# Fill missing values for numeric columns only
numeric_cols = data.select_dtypes(include=['number']).columns  # Select numeric columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Fill missing values with the mean

# Encode categorical variables (for simplicity, let's encode 'gender' and 'abonoment_type')
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['abonoment_type'] = label_encoder.fit_transform(data['abonoment_type'])

# Assuming 'id' is not a feature and 'name_personal_trainer' is not needed
features = data.drop(['id', 'name_personal_trainer', 'avg_time_check_in', 'avg_time_check_out'], axis=1)

# Convert features to float to ensure they are numeric
features = features.select_dtypes(include=['number'])  # Keep only numeric features

# Check if features DataFrame is empty
if features.empty:
    raise ValueError("No numeric features available after processing. Please check your dataset.")

target = data['abonoment_type']  # Let's assume we're predicting the type of subscription

# Convert features to PyTorch tensors (ensure they are numeric)
features_tensor = torch.tensor(features.values, dtype=torch.float32)
target_tensor = torch.tensor(target.values, dtype=torch.long)  # Use long for class labels

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Split dataset into two clients (simulating vertical split)
def split_dataset(features_tensor, target_tensor):
    mid = features_tensor.shape[1] // 2  # Split features into two halves
    client_1_data = CustomDataset(features_tensor[:, :mid], target_tensor)  # First half of features
    client_2_data = CustomDataset(features_tensor[:, mid:], target_tensor)  # Second half of features
    return client_1_data, client_2_data

client_1_data, client_2_data = split_dataset(features_tensor, target_tensor)

# Create DataLoaders for each client
client_1_loader = DataLoader(client_1_data, batch_size=32, shuffle=True)
client_2_loader = DataLoader(client_2_data, batch_size=32, shuffle=True)

# Define your model (example)
class ClientNet(nn.Module):
    def __init__(self, input_size):
        super(ClientNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, len(label_encoder.classes_))  # Number of classes for output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No activation here; use softmax in loss function for multi-class classification
        return x

# Initialize client models
input_size_client_1 = features.shape[1] // 2  # Half the number of features for client 1
input_size_client_2 = features.shape[1] - input_size_client_1  # Remaining features for client 2

client_model_1 = ClientNet(input_size_client_1)
client_model_2 = ClientNet(input_size_client_2)

# Example training function with FedProx regularization
def train_fedprox(client_model, client_loader, global_model=None, mu=0.01, epochs=1):
    optimizer = optim.SGD(client_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification

    for epoch in range(epochs):
        client_model.train()
        for data, target in client_loader:
            optimizer.zero_grad()
            output = client_model(data)

            loss = criterion(output, target)  # Cross-entropy loss
            
            # FedProx regularization term (if global model is provided)
            if global_model is not None:
                prox_loss = sum((p - g).norm(2) ** 2 for p, g in zip(client_model.parameters(), global_model.parameters()))
                loss += mu * prox_loss
            
            loss.backward()
            optimizer.step()

# Train each client model (this is just a placeholder example)
train_fedprox(client_model_1, client_1_loader)
train_fedprox(client_model_2, client_2_loader)

# Evaluation function (optional)
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)  # Get predicted class index
            total += target.size(0)
            correct += (predicted == target).sum().item()  # Compare with actual labels
    
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Evaluate each client's model separately (or combine them later)
evaluate_model(client_model_1, client_1_loader)
evaluate_model(client_model_2, client_2_loader)
