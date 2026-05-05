import os
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model.utils import load, ploter
from model.models import cnn_v1 , cnn_v2
from model.config import *

def build_model(model_ver:str):
    """Main function to build, train, and evaluate the CNN model for waste classification.

        Use v1 to test my custom CNN model
        Use v2 to test the transfer learning model using MobileNetV2 as a base
    """
    gpus = tf.config.experimental.list_physical_devices('GPU') #listing GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True) #limiting GPU memory growth
        
    load.validate_files(ROOT_DIR)
    val, train, test = load.process_data(ROOT_DIR, BATCH_SIZE)

    '''Custom CNN model'''
    if model_ver == "v1":
        label = f"""Blocks: {CONV_BLOCKS}\nConv Layers/Block {CONV_LAYERS}\nBatch Normalization: {BATCH_NORM}"""
        print(label)

        model = cnn_v1.Model(NUM_EPOCH) 
        built_model = model.build_model(CONV_BLOCKS, CONV_LAYERS, BATCH_NORM, DROPOUT)

    elif model_ver == "v2":
        '''Transfer learning model'''
        label = f"""Transfer Learning Model - MobileNetV2"""
        print(label)

        model = cnn_v2.Model(NUM_EPOCH) 
        built_model = model.build_model()
    else:
        print("Invalid model specified. Please choose 'v1' or 'v2'.")   
    
    hist = model.train_model(built_model, val, train, test) 
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    built_model.save(os.path.join(MODELS_DIR, f"waste_classification_model-{model_ver}.keras"))
    ploter.plot(hist,label)   

def run_model(model_ver:str):
    
    model = load_model(os.path.join(MODELS_DIR, f"waste_classification_model-{model_ver}.keras"))
    
        # Test on individual images
    print("\n=== Individual Image Predictions ===")
    if os.path.exists(TEST_DIR):
        for filename in os.listdir(TEST_DIR):
            img_path = os.path.join(TEST_DIR, filename)
            if not os.path.isfile(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load {filename}")
                continue
            
            # Resize and normalize
            resized = cv2.resize(img, (256, 256))
            normalized = resized / 255.0
            
            # Predict
            yhat = model.predict(np.expand_dims(normalized, axis=0), verbose=0)  # Shape: (1, 4)
            
            # Get the class with highest probability
            predicted_class = np.argmax(yhat[0])  # Get index of max probability
            confidence = np.max(yhat[0])  # Get the probability value
            prediction = CLASS_NAMES[predicted_class]
            
            print(f"Prediction for {filename}: {prediction} (confidence: {confidence:.2%})")
    else:
        print("'test_images' directory not found")
        
