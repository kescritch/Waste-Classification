import tensorflow as tf
import matplotlib.pyplot as plt
import cnn, load
from datetime import datetime
import numpy as np

def plot(hist:tf.keras.callbacks.History, label:str):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(bottom=0.3)  # make room for text below

    #loss plot
    axs[0].plot(hist.history['loss'], color='blue', label='loss')
    axs[0].plot(hist.history['val_loss'], color='orange', label='val_loss')
    axs[0].set_title('Loss', fontsize=20)
    axs[0].legend(loc="upper left")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epochs")

    #accuracy plot
    axs[1].plot(hist.history['accuracy'], color='blue', label='accuracy')
    axs[1].plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    axs[1].set_title('Accuracy', fontsize=20)
    axs[1].legend(loc="upper left")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epochs")

    max_acc = max(hist.history['accuracy'])
    max_val_acc = max(hist.history['val_accuracy'])
    min_val_loss = min(hist.history['val_loss'])
    min_loss = min(hist.history['loss'])

    plt.subplots_adjust(bottom=0.30)

    plt.figtext(0.25, 0.08, label, ha='center', fontsize=9)
    plt.figtext(0.75, 0.08,
                f"Max acc: {max_acc:.4f}  |  Max val acc: {max_val_acc:.4f}\nMin loss: {min_loss:.4f}  |  Min val loss: {min_val_loss:.4f}",
                ha='center', fontsize=9)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    name = f"loss_plot_and_accuracy_plot_{date_str}_{time_str}.png"
    plt.savefig(name, bbox_inches='tight')

ROOT_DIR = 'training_data'
BATCH_SIZE = 32
NUM_EPOCH = 40

CONV_BLOCKS = 4
CONV_LAYERS = 1
NUM_DENSE = 2
BATCH_NORM = True
DROPOUT = False

def main():
    
    gpus = tf.config.experimental.list_physical_devices('GPU') #listing GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True) #limiting GPU memory growth
        
    load.validate_files(ROOT_DIR)
    val, train, test = load.process_data(ROOT_DIR, BATCH_SIZE)

    label = f"""Blocks: {CONV_BLOCKS}\nConv Layers/Block {CONV_LAYERS}\nBatch Normalization: {BATCH_NORM}"""
    print(label)
    model = cnn.Model(NUM_EPOCH) 
    built_model = model.build_model(CONV_BLOCKS, CONV_LAYERS, BATCH_NORM, DROPOUT)
    hist = model.train_model(built_model, val, train, test) 
    plot(hist,label)    
    
if __name__ == '__main__':
    main()