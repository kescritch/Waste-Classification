from xml.parsers.expat import model

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report

import os

class Model:
    '''
    CNN model builder and trainer for waste classification using transfer learning with MobileNetV2.
    '''
    def __init__(self,epochs:int = 20):
        self.epochs = epochs


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
        model.add(Dropout(.2))
        model.add(Dense(128, activation = 'relu')) 
        model.add(Dropout(.2))
        model.add(Dense(4, activation = 'softmax')) 

        model.compile(Adam(learning_rate=1e-3), loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
        
        return model


    def train_model(self,model:Sequential, val:tf.data.Dataset, train:tf.data.Dataset, test:tf.data.Dataset):
        '''
            Trains the model and prints classification report on test data.
        '''
        
        logdir = 'logs'
        callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        
        hist = model.fit(train, epochs = self.epochs, validation_data = val, callbacks = [callback])

        #printing classification accuracy
        y_true = []
        y_pred = []

        for images, labels in test:
            predictions = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.argmax(axis=1))

        print(classification_report(y_true, y_pred,
            target_names=['not_allowed', 'paper', 'plastic', 'trash']))
        return hist

def main(): 
    model = Model(10)
    results = model.build_model()
    results.summary()

    
if __name__ == "__main__":
    main()