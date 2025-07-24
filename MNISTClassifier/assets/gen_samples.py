from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='../../Datasets/MNIST', train=True, download=True, transform=transform)

for i, (image, label) in enumerate(mnist_data):
    image_pil = transforms.ToPILImage()(image)
    image_path = f'{label}_{i}.png'
    image_pil.save(image_path)
    if i == 10:
        break
