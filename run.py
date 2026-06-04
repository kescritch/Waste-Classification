import app.main as model
import app.camera.camera_runner as camera
import app.utils.calibrate_background as calibrate
import app.utils.image_processor as image_processor
import os
from app.config import ROOT_DIR, MODELS_DIR
image_processor.get_and_print_distribution(ROOT_DIR)
# model_ver = "v2" # Choose 'v1' for custom CNN or 'v2' for transfer learning with MobileNetV2

# model.main(model_ver) #builds the model

# calibrate.run_calibration()
# built_model = model.build_model(model_ver)

#use already existing model

# model_path = os.path.join(MODELS_DIR, f"waste_classification_model-{model_ver}.keras")
# built_model = model.load(model_path)

# model.test_model(built_model)
# camera.run_camera_with_model(built_model)