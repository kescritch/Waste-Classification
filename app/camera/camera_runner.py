import cv2
import numpy as np
import os
from app.config import MODELS_DIR, CLASS_NAMES

def run_camera_with_model(model):
    camera = cv2.VideoCapture(0)
    
    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to capture image")
            break
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4  # center box

        # Crop and classify just the box region
        crop = image[y1:y2, x1:x2]
        resized = cv2.resize(crop, (254, 254))
        normalized = resized / 255.0
        
        yhat = model.predict(np.expand_dims(normalized, axis=0), verbose=0)
        predicted_class = np.argmax(yhat[0])
        confidence = np.max(yhat[0])
        prediction = CLASS_NAMES[predicted_class]

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if prediction == "none":
            label = ""
        else:
            label = f"{prediction} ({confidence:.2%})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Waste Classification', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()