import torch
from torch import device

import os

from app.utils import image_processor, model_helper, data_analysis, camera_runner
from app.models.cnn_v1 import CNN_V1
from app.models.cnn_v2 import CNN_V2
from app.config import *

def build_model(model_ver:str, device = "cpu") -> None:
    """Main function to build, train, and evaluate the CNN model for waste classification.

        Use v1 to test my custom CNN model
        Use v2 to test the transfer learning model using MobileNetV2 as a base
    """
    
    print("Building model...")
    if model_ver == "v1":
        model = CNN_V1(num_classes=len(CLASS_NAMES)).to(device)
    elif model_ver == "v2":
        model = CNN_V2(num_classes=len(CLASS_NAMES)).to(device)
    print(model)
    
    train_loader, val_loader, test_loader = image_processor.process_data(ROOT_DIR, batch_size=BATCH_SIZE)
    
    print("Training model...")
    hist = model_helper.train_model(model, train_loader, val_loader, NUM_EPOCHS, device)
   
    print("Evaluating model...")
    model_helper.evaluate_model(model, test_loader, device)
    data_analysis.plot(hist, f"Model version: {model_ver} | Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE}")
    model_helper.image_test(model,TEST_DIR, device)
    
    print("Saving model...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"waste_classification_model-{model_ver}.pth")
    torch.save(model.state_dict(), model_path)

def run_model(model_ver:str, device = "cpu") -> None:
    """
    Loads a saved model based on the version and runs it on the webcam feed.
    
    Use v1 to test my custom CNN model.
    Use v2 to test the transfer learning model using MobileNetV2 as a base.
    """
    print(f"Using device: {device}")
    
    model = model_helper.load_model(model_ver)
    model.to(device)
    
    camera_runner.video(model, device)

