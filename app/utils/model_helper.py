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
            data    = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            scores  = model(data)
            loss    = criterion(scores, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Loss: {running_loss / len(train_loader):.4f}")




def evaluate_model(model, test_loader, device):
    
    acc = Accuracy(task="multiclass", num_classes=len(CLASS_NAMES))
    precision = Precision(task="multiclass", num_classes=len(CLASS_NAMES), average="macro")
    recall = Recall(task="multiclass", num_classes=len(CLASS_NAMES), average="macro")
    
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
                        