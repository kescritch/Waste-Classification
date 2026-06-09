from PIL import Image as PILImage
import torch
from torch import device

import os

from torchvision import transforms

from app.utils import image_processor, model_helper, data_analysis
from app.models.cnn_v1 import CNN_V1
from app.models.cnn_v2 import CNN_V2
from app.config import *

def build_model(model_ver:str):
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

def classify_test_images(model_ver: str = "v1"):
    device = "cpu"
    model = model_helper.load_model(model_ver)

    # Print index -> class mappings
    print("\n--- Class Index Mappings ---")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  Index {idx} -> {name}")
    print("----------------------------\n")

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Gather all images from TEST_DIR
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"No images found in {TEST_DIR}")
        return

    print(f"Found {len(image_files)} image(s) in {TEST_DIR}\n")

    for filename in image_files:
        img_path = os.path.join(TEST_DIR, filename)
        
        try:
            img = PILImage.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {filename} — could not open: {e}")
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item() * 100

        predicted_class = CLASS_NAMES[predicted_idx]
        print(f"  {filename}")
        print(f"    Predicted : {predicted_class} (index {predicted_idx})")
        print(f"    Confidence: {confidence:.2f}%\n")

