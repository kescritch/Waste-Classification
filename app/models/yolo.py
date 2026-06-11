import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

class WasteDetector:
    def __init__(self, classifier_model, class_names, device="cpu"):
        self.yolo = YOLO("yolov8n.pt")
        self.classifier = classifier_model.to(device)
        self.classifier.eval()
        self.class_names = class_names
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        # frame: BGR numpy array from OpenCV
        results = self.yolo(frame, verbose=False, device=self.device)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf < 0.3:  # filter weak detections
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            print(f"Cropshape {crop.shape}, , aspect ratio (w/h): {crop.shape[1]/crop.shape[0]:.2f}")

            # Convert BGR -> RGB -> PIL -> tensor
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.classifier(tensor)
                pred_idx = output.argmax(dim=1).item()
                pred_class = self.class_names[pred_idx]
                pred_conf = torch.softmax(output, dim=1)[0, pred_idx].item()

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "yolo_conf": conf,
                "class": pred_class,
                "class_conf": pred_conf
            })

        return detections

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['class_conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame