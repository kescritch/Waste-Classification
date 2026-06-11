import os

# Paths
ROOT_DIR = 'app/data/training_data/archive'
TEST_DIR = 'app/data/test_images'

OUTPUT_DIR = 'app/output'
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')


#Training Params
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.001

# Model Params
CLASS_NAMES = ["cardboard-paper", "glass", "metal", "plastic"]
CONV_BLOCKS = 4
CONV_LAYERS = 1
NUM_DENSE = 2
BATCH_NORM = True
DROPOUT = 0.5 #not implemented yet

# Create directories if they don't exist
# for directory in [OUTPUT_DIR, LOGS_DIR, PLOTS_DIR, MODELS_DIR]:
#     os.makedirs(directory, exist_ok=True)
    
print("models directory:", MODELS_DIR)
