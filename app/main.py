import torch
from torch import device

import os
import cv2
import numpy as np

from app.utils import image_processor, model_helper
from app.models.cnn_v1 import CNN_V1
from app.models.cnn_v2 import CNN_V2
from app.config import *

def main(model_ver:str):
    """Main function to build, train, and evaluate the CNN model for waste classification.

        Use v1 to test my custom CNN model
        Use v2 to test the transfer learning model using MobileNetV2 as a base
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Building model...")
    if model_ver == "v1":
        model = CNN_V1(num_classes=len(CLASS_NAMES)).to(device)
    elif model_ver == "v2":
        model = CNN_V2(num_classes=len(CLASS_NAMES)).to(device)
    print(model)
    
    val_loader, train_loader, test_loader = image_processor.process_data(ROOT_DIR, batch_size=BATCH_SIZE)
    
    print("Training model...")
    model_helper.train_model(model, train_loader, val_loader, NUM_EPOCHS, device)
   
    print("Evaluating model...")
    model_helper.evaluate_model(model, test_loader, device)
    
    model_helper.image_test(model,TEST_DIR, device)
    
    print("Saving model...")
    model_path = os.path.join(MODELS_DIR, f"waste_classification_model-{model_ver}.pth")
    torch.save(model.state_dict(), model_path)

def load(path):   
    return load_model(path)
