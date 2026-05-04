from xml.parsers.expat import model
import cv2
from matplotlib.pylab import test
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import classification_report
import numpy as np
import os

class Model:
    '''
    CNN model builder and trainer for waste classification using transfer learning with MobileNetV2.
    '''
    def __init__(self,epochs:int = 20):
        self.epochs = epochs
        self.class_names = [
            'not_allowed',
            'paper',
            'plastic',
            'trash'
        ]

    def build_model(self) -> Sequential:
        '''
            Builds the model using MobileNetV2 as a base and adds fully connected layers on top.
        '''

        image_shape = (256, 256, 3)
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',  # Load pre-trained ImageNet weights
            include_top=False,   # Exclude the top classification layer
        )

        base_model.trainable = False # Freeze the base model

        model = Sequential([])
        model.add(tf.keras.Input(image_shape))
        model.add(base_model) # Add the base model as a feature extractor

        #fully connected layers
        model.add(GlobalAveragePooling2D()) #flattens the channel value from the last block into one

        model.add(Dense(128, activation = 'relu')) 

        model.add(Dense(4, activation = 'softmax')) 

        model.compile(Adam(learning_rate=1e-3), loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
        
        return model


    def train_model(self, model: Sequential, val: tf.data.Dataset, train: tf.data.Dataset, test: tf.data.Dataset):
        '''
        Trains the model and prints classification report on test data
        '''
        logdir = 'logs'
        callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        
        # Train the model
        hist = model.fit(train, epochs=self.epochs, validation_data=val, callbacks=[callback])
        
        # Initialize metrics for 4-class classification
        precision = Precision()
        recall = Recall()
        
        all_y_true = []
        all_y_pred = []
        
        # Evaluate on test set
        for batch in test.as_numpy_iterator():
            X, y = batch
            y_pred = model.predict(X, verbose=0)  # Shape: (batch_size, 4)
            
            # Convert model output (probabilities) to class indices
            y_pred_classes = tf.argmax(y_pred, axis=1).numpy()  # Shape: (batch_size,)
            
            all_y_true.extend(y)  # y is already class indices from your dataset
            all_y_pred.extend(y_pred_classes)
            
            # Update metrics with class indices
            precision.update_state(y, y_pred_classes)
            recall.update_state(y, y_pred_classes)
        
        # Convert to numpy arrays
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        print("\n=== Test Set Performance ===")
        print(f"Precision: {precision.result().numpy():.4f}")
        print(f"Recall: {recall.result().numpy():.4f}")
        print("\n=== Classification Report ===")
        print(classification_report(all_y_true, all_y_pred, target_names=self.class_names))
        
        # Test on individual images
        print("\n=== Individual Image Predictions ===")
        if os.path.exists('test_images'):
            for filename in os.listdir('test_images'):
                img_path = os.path.join('test_images', filename)
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
                prediction = self.class_names[predicted_class]
                
                print(f"Prediction for {filename}: {prediction} (confidence: {confidence:.2%})")
        else:
            print("'test_images' directory not found")
        
        return hist

def main(): 
    model = Model(10)
    results = model.build_model()
    results.summary()

    
if __name__ == "__main__":
    main()