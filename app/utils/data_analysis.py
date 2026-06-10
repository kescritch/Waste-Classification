from datetime import datetime
import os
import matplotlib.pyplot as plt
from app.config import PLOTS_DIR


def plot(hist: dict, label: str) -> None:
    """Plots the training and validation loss and accuracy from the training history."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Loss plot
    axs[0].plot(hist['loss'],     color='blue',   label='loss')
    axs[0].plot(hist['val_loss'], color='orange', label='val_loss')
    axs[0].set_title('Loss', fontsize=20)
    axs[0].legend(loc="upper left")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epochs")

    # Accuracy plot
    axs[1].plot(hist['accuracy'],     color='blue',   label='accuracy')
    axs[1].plot(hist['val_accuracy'], color='orange', label='val_accuracy')
    axs[1].set_title('Accuracy', fontsize=20)
    axs[1].legend(loc="upper left")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epochs")

    # Summary text
    max_acc      = max(hist['accuracy'])
    max_val_acc  = max(hist['val_accuracy'])
    min_loss     = min(hist['loss'])
    min_val_loss = min(hist['val_loss'])

    plt.subplots_adjust(bottom=0.30)
    plt.figtext(0.25, 0.08, label, ha='center', fontsize=9)
    plt.figtext(0.75, 0.08,
                f"Max acc: {max_acc:.4f}  |  Max val acc: {max_val_acc:.4f}\n"
                f"Min loss: {min_loss:.4f}  |  Min val loss: {min_val_loss:.4f}",
                ha='center', fontsize=9)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    name = f"loss_plot_and_accuracy_plot_{date_str}_{time_str}.png"
    plt.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')
    plt.close(fig)