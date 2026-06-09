import os 
import torch
from tqdm import tqdm
from app.config import *

from torchmetrics.classification import Accuracy, Precision, Recall

from app.models.cnn_v1 import CNN_V1
from app.models.cnn_v2 import CNN_V2

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    hist = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        running_loss  = 0.0
        correct       = 0
        total         = 0
        for batch_index, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
            data    = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            scores  = model(data)
            loss    = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted  = torch.max(scores, 1)
            correct      += (predicted == targets).sum().item()
            total        += targets.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total
        hist['loss'].append(train_loss)
        hist['accuracy'].append(train_acc)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        # --- Validation (every 5 epochs and on the last epoch) ---
        if epoch == num_epochs - 1:
            model.eval()
            val_loss     = 0.0
            val_correct  = 0
            val_total    = 0
            with torch.no_grad():
                for data, targets in val_loader:
                    data    = data.to(device)
                    targets = targets.to(device)
                    outputs = model(data)
                    loss    = criterion(outputs, targets)
                    val_loss    += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == targets).sum().item()
                    val_total   += targets.size(0)

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc  = val_correct / val_total
            print(f"Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {epoch_val_acc:.4f}")
        else:
            # Carry forward last known val metrics so hist lists stay the same length as loss/accuracy
            epoch_val_loss = hist['val_loss'][-1]  if hist['val_loss']  else 0.0
            epoch_val_acc  = hist['val_accuracy'][-1] if hist['val_accuracy'] else 0.0

        hist['val_loss'].append(epoch_val_loss)
        hist['val_accuracy'].append(epoch_val_acc)

    return hist


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


def load_model(model_ver: str):
    """Loads a saved model from disk."""
    device = "cpu"
    print(f"Using device: {device}")
    
    if model_ver == "v1":
        model = CNN_V1(num_classes=len(CLASS_NAMES)).to(device)
    elif model_ver == "v2":
        model = CNN_V2(num_classes=len(CLASS_NAMES)).to(device)
    
    model_path = os.path.join(MODELS_DIR, f"waste_classification_model-{model_ver}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model