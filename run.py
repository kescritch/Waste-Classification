import app.main as model
from app.utils import image_processor

"""
    To run a saved model uncomment model.run_model and comment out model.build model with the appropriate model version.
    To save a new model uncomment model.build_model and comment out model.run_model.
"""
   
model_ver = "v2" # Choose 'v1' for custom CNN or 'v2' for transfer learning with MobileNetV2

#---------------------------------------------------------------------------------------------#

#image_processor.get_and_print_distribution("app/data/training_data/archive")

# model.build_model(model_ver, "cuda") #builds the model

#---------------------------------------------------------------------------------------------#

model.run_model(model_ver) #classifies the test images in the test_images folderpip install matplotlib
