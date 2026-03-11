import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os


def build_model() -> Sequential:

    model = Sequential() #initializing class

    #adding layers 
    model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (256,256,3))) #convolution with 16 filters with the size of 3 pixels by 3 pixels. Stride of 1
    model.add(MaxPooling2D()) #gets the maximum value from the relu, reduces the image data 

    model.add(Conv2D(32, (3,3), 1, activation = 'relu')) #convolution with 32 filters
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3,3), 1, activation = 'relu')) #convolution with 16 filters
    model.add(MaxPooling2D())

    model.add(Flatten()) #flattens the channel value from the last block into one

    model.add(Dense(256, activation='relu')) 
    model.add(Dense(4, activation = 'softmax'))
    
    model.compile('adam', loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
    
    return model

def train_model(model : Sequential, val: tf.data.Dataset, train: tf.data.Dataset, test: tf.data.Dataset):
    logdir = 'logs'
    callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    hist = model.fit(train, epochs = 20, validation_data=val, callbacks=[callback])

    return hist

def main(): 
    model = build_model()
    model.summary()
    
    model.train(model)
    
if __name__ == "__main__":
    main()