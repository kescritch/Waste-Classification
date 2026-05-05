import cv2
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, regularizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from sklearn.metrics import classification_report
import numpy as np
import os

from model.config import LOGS_DIR, TEST_DIR, CLASS_NAMES

class Model:
    """
    CNN model builder and trainer for waste classification.
    Supports configurable convolutional blocks, batch norm, and dropout.
    """
    def __init__(self,epochs:int = 20):
        self.epochs = epochs
        self.class_names = CLASS_NAMES
        

    def block(self,model:int, mult:int, batch_norm:bool, conv_layers:int): 
        '''
            Convolutional block model. 
            Each block has 1 or 2 convolutional layers, followed by optional batch normalization, relu activation, and max pooling. 
            The number of filters doubles with each block.
        '''
        filters = 32 * (2 ** mult) #doubles the number of filters with each block

        model.add(Conv2D(filters, (3,3), 1, padding = 'same')) #convolution with 32 filters with the size of 3 pixels by 3 pixels. Stride of 1
        if conv_layers == 2:
            model.add(Conv2D(filters, (3,3), 1, padding = 'same')) #adding 2nd conv layer if specified

        if batch_norm:  
            model.add(BatchNormalization())

        model.add(Activation('relu')) #activation function
        model.add(MaxPooling2D(pool_size = (2,2), strides=2))

    def build_model(self,blocks:int, conv_layers:int, batch_norm:bool, dropout:bool) -> Sequential:
        '''
            Builds the model.
            blocks: number of convolutional blocks
            conv_layers: number of convolutional layers in each block (1 or 2)
            batch_norm: whether to use batch normalization
            dropout: whether to use dropout in the fully connected layers
        '''

        model = Sequential()
        model.add(tf.keras.Input(shape=(256, 256, 3)))
        
        for i in range(blocks): #adding blocks 
            self.block(model, i, batch_norm, conv_layers)

        #fully connected layers
        model.add(GlobalAveragePooling2D()) #flattens the channel value from the last block into one
        if dropout: 
            model.add(Dropout(.2))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        if dropout:
            model.add(Dropout(.2))
        model.add(Dense(4, activation = 'softmax')) 

        model.compile(Adam(learning_rate=1e-3), loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
        
        return model

    def train_model(self, model: Sequential, val: tf.data.Dataset, train: tf.data.Dataset, test: tf.data.Dataset):
        '''
        Trains the model and prints classification report on test data
        '''
        logdir = LOGS_DIR
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
                prediction = self.class_names[predicted_class]
                
                print(f"Prediction for {filename}: {prediction} (confidence: {confidence:.2%})")
        else:
            print("'test_images' directory not found")
        
        return hist

def main(): 
    model = Model(10)
    results = model.build_model(4,1,True,False)
    results.summary()

    
if __name__ == "__main__":
    main()