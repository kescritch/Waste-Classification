import app.main as model
import app.camera.camera_runner as camera
import app.utils.calibrate_background as calibrate
import app.utils.image_processor as image_processor
import os



"""
    To run a saved model uncomment model.run_model and comment out model.build model with the appropriate model version.
    To save a new model uncomment model.build_model and comment out model.run_model.
"""
    
model_ver = "v2" # Choose 'v1' for custom CNN or 'v2' for transfer learning with MobileNetV2

#---------------------------------------------------------------------------------------------#

# model.build_model(model_ver) #builds the model

#---------------------------------------------------------------------------------------------#

# model.run_model(model_ver) #classifies the test images in the test_images folderpip install matplotlib