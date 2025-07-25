from model import MNISTClassifier, nn
from torch import optim
from data import train_loader, test_loader
from os import makedirs
import torch

model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test acc: {100 * correct / total:.2f}%')


def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}]. Train loss: {running_loss / len(train_loader):.4f}')
        test(model, test_loader)


if __name__ == '__main__':
    makedirs('checkpoints', exist_ok=True)
    train(model, train_loader, criterion, optimizer, epochs=20)
    torch.save(model.state_dict(), 'checkpoints/mnist_classifier.pth')
