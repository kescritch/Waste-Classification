import cv2
import os
from app.config import ROOT_DIR

def run_calibration():
    camera = cv2.VideoCapture(0)
    count = 0
    save_dir = os.path.join(ROOT_DIR, "none")
    os.makedirs(save_dir, exist_ok=True)

    while count < 200:
        ret, frame = camera.read()
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4  # center box

        # Crop and classify just the box region
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (254, 254))
        normalized = resized / 255.0
            
        cv2.imshow("Capture", resized)
        
        key = cv2.waitKey(1)
        cv2.imwrite(f"{save_dir}/none_{count}.jpg", resized)
        count += 1
        print(f"Saved {count}")


    camera.release()
    cv2.destroyAllWindows()