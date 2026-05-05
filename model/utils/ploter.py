from datetime import datetime
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from model.config import PLOTS_DIR


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

    #printing other
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
    plt.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')


