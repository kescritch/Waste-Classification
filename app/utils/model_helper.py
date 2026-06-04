import torch
from tqdm import tqdm
from app.config import *

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