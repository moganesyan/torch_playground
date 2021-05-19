import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Initialise NN architecture
class NN(nn.Module):
    # NN class initialisation
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    # feedforward step
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Initialize datasets and model
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

model = NN(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device = device)

        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, targets)

        # reset gradients for each batch before stepping back
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print(f"Checking accuracy for training data")
    else:
        print(f"Checking accuracy for testing data")

    num_correct = 0
    num_samples = 0

    # set model to evaluation mode
    model.eval()

    # dont compute gradients in evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct / num_samples) * 100:.2f}")

    # change model back to training mode
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)





