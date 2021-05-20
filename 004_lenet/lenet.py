import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Initialise Lenet architecture
class Lenet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = (5,5),
            stride = (1,1), padding = (0,0))
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5,5),
            stride = (1,1), padding = (0,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5,5),
            stride = (1,1), padding = (0,0))
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# transforms to make MNISt images compatible with lenet
my_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2, fill = 0, padding_mode = 'constant'),
    transforms.ToTensor()
    ])

# checkpoint helper functions
def save_checkpoint(state, filename = 'my_checkpoint.pth'):
    print('Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print('Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
load_model = False

# Initialize datasets and model
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = my_transforms, download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = my_transforms, download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

model = Lenet(in_channels = in_channels, num_classes = num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth"))
# Training loop
for epoch in range(num_epochs):
    print(f"Doing epoch {epoch}")
    losses = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device = device)

        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

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

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct / num_samples) * 100:.2f}")

    # change model back to training mode
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)





