from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='../Datasets/MNIST', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../Datasets/MNIST', train=False, download=True, transform=transform)

indices = list(range(1000))  # choose 1000 samples
subset_dataset = Subset(train_dataset, indices)

train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(subset_dataset, batch_size=1000, shuffle=False)
