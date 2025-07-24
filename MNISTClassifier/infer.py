import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageTk
from model import MNISTClassifier
import tkinter as tk

model = MNISTClassifier()
model.load_state_dict(torch.load('checkpoints/mnist_classifier.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_and_predict(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_label = torch.max(output, 1)
    return image, predicted_label.item()


def display_image_and_label(image_path, true_label):
    image, predicted_label = load_and_predict(image_path)

    root = tk.Tk()
    root.title("Inference")

    image_tk = ImageTk.PhotoImage(image)

    img_label = tk.Label(root, image=image_tk)
    img_label.pack()

    result_label = tk.Label(root, text=f"True Label: {true_label}\nPredicted Label: {predicted_label}")
    result_label.pack()

    root.mainloop()


image_path = Path('assets/1_3.png')
true_label = image_path.name.split('_')[0]

display_image_and_label(image_path, true_label)
