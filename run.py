import app.main as model
import app.camera.camera_runner as camera
import app.utils.calibrate_background as calibrate
import app.utils.image_processor as image_processor
import os

model_ver = "v2" # Choose 'v1' for custom CNN or 'v2' for transfer learning with MobileNetV2

model.build_model(model_ver) #builds the model

#---------------------------------------------------------------------------------------------#

# model.classify_test_images(model_ver) #classifies the test images in the test_images folderpip install matplotlib