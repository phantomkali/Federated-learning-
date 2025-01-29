import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Neural Network for vertical federated learning (each client)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 // 2, 128)  # Half the feature size (392)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Split dataset into two clients, each with half of the features
def split_features(data):
    half_size = data.size(1) // 2
    return data[:, :half_size], data[:, half_size:]

# Training function for each client
def train_client(client_data, client_id):
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(client_data, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(1):
        for data, target in train_loader:
            optimizer.zero_grad()
            data_left, data_right = split_features(data.view(-1, 28 * 28))

            if client_id == 0:
                client_data = data_left
            else:
                client_data = data_right

            output = model(client_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()

# Federated averaging
global_model_left = Net()
global_model_right = Net()
global_weights_left = None
global_weights_right = None
num_clients = 2
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

for client_id in range(num_clients):
    client_weights = train_client(train_dataset, client_id)

    if client_id == 0:
        global_weights_left = client_weights
    else:
        global_weights_right = client_weights

# Average the weights for both models (left and right)
for key in global_weights_left.keys():
    global_weights_left[key] /= num_clients
    global_weights_right[key] /= num_clients

global_model_left.load_state_dict(global_weights_left)
global_model_right.load_state_dict(global_weights_right)

# Evaluate the model by combining both sides (left and right features)
def evaluate_combined_model():
    global_model_left.eval()
    global_model_right.eval()

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=32)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data_left, data_right = split_features(data.view(-1, 28 * 28))
            
            output_left = global_model_left(data_left)
            output_right = global_model_right(data_right)
            
            # Combine the outputs (simple averaging or addition)
            combined_output = (output_left + output_right) / 2
            
            _, predicted = torch.max(combined_output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# Call the function to evaluate the combined model
evaluate_combined_model()
