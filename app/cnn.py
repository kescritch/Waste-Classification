import tensorflow as tf
from tensorflow import keras
from keras import Sequential, regularizers
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report

import os

class Model:
    def __init__(self,epochs:int = 20):
        self.epochs = epochs
        

    def block(self,model:int, mult:int, first_block:bool, batch_norm:bool, conv_layers:int): #testing how different variables effect overfitting
        filters = 32 * (2 ** mult) #doubles the number of filters with each block
        print(f"Filter size: {filters}")

        if(first_block):
            model.add(Conv2D(filters, (3,3), 1, padding = 'same')) #convolution with 32 filters with the size of 3 pixels by 3 pixels. Stride of 1
            if conv_layers == 2:
                model.add(Conv2D(filters, (3,3), 1, padding = 'same'))
        else: 
            model.add(Conv2D(filters, (3,3), 1, padding = 'same'))
            if conv_layers == 2:
                model.add(Conv2D(filters, (3,3), 1, padding = 'same'))
        if batch_norm:
            model.add(BatchNormalization())

        model.add(Activation('relu')) 
        model.add(MaxPooling2D(pool_size = (2,2), strides=2)) #gets the maximum value from the relu, reduces the image data

    def build_model(self,blocks:int, conv_layers:int, batch_norm:bool, dropout:bool) -> Sequential:
        
        model = Sequential()
        model.add(tf.keras.Input(shape=(256, 256, 3)))
        
        for i in range(blocks): #adding blocks 
            self.block(model, i, i==0, batch_norm, conv_layers)

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

    def train_model(self,model:Sequential, val:tf.data.Dataset, train:tf.data.Dataset, test:tf.data.Dataset):
        # Reduce learning rate when val_loss stops improving
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',   # which metric to monitor
            factor=0.5,           # reduce LR by this factor
            patience=3,           # wait this many epochs without improvement
            min_lr=1e-6,          # don't go below this LR
            verbose=1
        )

        #Adding weights because of under represented data
        # class_weights = {
        #     0: 5.07,  # not_allowed (1239 images)
        #     1: 1.99,  # paper (3163 images)
        #     2: 2.39,  # plastic (2630 images)
        #     3: 1.0    # trash (6279 images)
        # }
        
        logdir = 'logs'
        callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        
        hist = model.fit(train, epochs = self.epochs, validation_data = val, callbacks = [callback,reduce_lr])

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
    results = model.build_model(4,1,True,False)
    results.summary()

    
if __name__ == "__main__":
    main()