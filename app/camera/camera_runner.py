import cv2
import numpy as np
import torch
from torchvision import transforms
from app.config import CLASS_NAMES

def run_camera_with_model(model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.eval()
    camera = cv2.VideoCapture(0)

    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to capture image")
            break

        h, w = image.shape[:2]
        x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4

        # Crop and preprocess
        crop = image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        prediction = CLASS_NAMES[predicted.item()]
        confidence  = confidence.item()

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "" if prediction == "none" else f"{prediction} ({confidence:.2%})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Waste Classification', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()