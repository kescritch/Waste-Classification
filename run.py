import app.main as model
from app.utils import image_processor

"""
To run a saved model uncomment model.run_model and comment out model.build_model with the appropriate model version.
To save a new model uncomment model.build_model and comment out model.run_model.
"""

model_ver = "v2"

# image_processor.get_and_print_distribution("app/data/training_data/archive")

model.build_model(model_ver, "cuda")

# model.run_model(model_ver)

