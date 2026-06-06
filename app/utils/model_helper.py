import os 
import torch
from tqdm import tqdm
from app.config import *

from torchmetrics.classification import Accuracy, Precision, Recall


def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_index, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Loss: {running_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader, device):
    acc = Accuracy(task="multiclass", num_classes=len(CLASS_NAMES)).to(device)
    precision = Precision(task="multiclass", num_classes=len(CLASS_NAMES), average="macro").to(device)
    recall = Recall(task="multiclass", num_classes=len(CLASS_NAMES), average="macro").to(device)

    model.eval()

    with torch.no_grad():
        for image, target in test_loader:
            image = image.to(device)
            target = target.to(device)

            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

            acc(predicted, target)
            precision(predicted, target)
            recall(predicted, target)

    print(f"Test Accuracy: {acc.compute().item():.4f}\n"
          f"Test Precision: {precision.compute().item():.4f}\n"
          f"Test Recall: {recall.compute().item():.4f}")


def image_test(model, image_path, device):
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    dir_path = os.path.dirname(image_path) or '.'
    for filename in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, filename)

        if not os.path.isfile(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        model.eval()

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        print(f"File: {filename} - Predicted class: {CLASS_NAMES[predicted.item()]}")


def load_model(path):
    return torch.load(path)  
