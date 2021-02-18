import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    output = F.relu(self.conv1(x))            # Output size: N x 6 x 24 x 24
    output = self.pool(output)                # Output size: N x 6 x 12 x 12
    output = F.relu(self.conv2(output))       # Output size: N x 16 x 8 x 8
    output = self.pool(output)                # Output size: N x 16 x 4 x 4
    output = output.view(output.shape[0], -1) # Output size: N x 256
    output = F.relu(self.fc1(output))         # Output size: N x 120
    output = F.relu(self.fc2(output))         # Output size: N x 84
    output = self.fc3(output)                 # Output size: N x 10
    return output


def main():

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"[INFO] Using GPU: {torch.cuda.is_available()}")
    print(f"[INFO] Device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    # Load data
    print("[INFO] Loading datasets ...")
    data_dir = "data/"
    # Training data
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # Testing data
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    print("[INFO] Compiling model ...")
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("[INFO] Training model ...")
    for epoch in range(20):

        # Training step
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Zero gradient used by optimizer
            optimizer.zero_grad()
            # Predict outputs
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch:05d} [{batch_idx * len(inputs):05d}/{len(train_loader.dataset)}] -- Loss: {loss.item()}")

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Predict outputs
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct += predictions.eq(targets.view_as(predictions)).sum().item()
        print(f"Accuracy: {correct / len(test_loader.dataset)}")

    torch.save(model.state_dict(), "model.pt")











if __name__ == '__main__':
    main()
