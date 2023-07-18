import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Define a simple MLP model
class AgePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AgePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Define a simple training dataset
class AgeDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.targets[index], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.data)


# Define your training data and targets
train_data = [[1.2, 3.4, 5.6], [2.1, 4.3, 6.5], [3.4, 5.6, 7.8], [4.5, 6.7, 8.9]]
train_targets = [20.0, 25.0, 30.0, 35.0]

# Create a training dataset and pass it to the DataLoader function
train_dataset = AgeDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Define the model, loss function, and optimizer
model = AgePredictor(input_size=3, hidden_size=4, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Ask questions to predict the user's age
print('Answer the following questions to predict your age')
height = float(input('Enter your height (in cm): '))
weight = float(input('Enter your weight (in kg): '))
gender = int(input('Enter your gender (0 for male, 1 for female): '))

# Normalize the inputs
inputs = [(height - 170) / 10, (weight - 70) / 10, gender]

# Convert the inputs to a tensor and pass them to the model
inputs = torch.tensor(inputs, dtype=torch.float32)
age = model(inputs).item()

# Print the predicted age
print(f'Your predicted age is: {age:.1f}')

# Save the trained model to a file
torch.save(model.state_dict(), 'age_predictor.pth')

# model = AgePredictor(input_size=3, hidden_size=4, output_size=1)
# model.load_state_dict(torch.load('age_predictor.pth'))