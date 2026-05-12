from app.config import MODELS_DIR
import app.main as model
import app.camera.camera_runner as camera
import app.utils.calibrate_background as calibrate
import os

model_ver = "v2" # Choose 'v1' for custom CNN or 'v2' for transfer learning with MobileNetV2

# train new model

# calibrate.run_calibration()
# built_model = model.build_model(model_ver)

#use already existing model

model_path = os.path.join(MODELS_DIR, f"waste_classification_model-{model_ver}.keras")
built_model = model.load(model_path)

model.test_model(built_model)
camera.run_camera_with_model(built_model)