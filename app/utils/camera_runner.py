import cv2
from app.config import CLASS_NAMES
from app.models.yolo import WasteDetector

def video(model, device) -> None:
    detector = WasteDetector(model, CLASS_NAMES, device)
    camera = cv2.VideoCapture(0)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        detections = detector.process_frame(frame)
        frame = detector.draw_detections(frame, detections)
        
        cv2.imshow("Waste Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()