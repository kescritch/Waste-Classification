import tensorflow as tf
import matplotlib.pyplot as plt
import cnn, load
from datetime import datetime

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
        
    root_dir = 'training_data'
    
    load.validate_files(root_dir)
    val, train, test = load.process_data(root_dir)
    
    model = cnn.build_model()
    hist = cnn.train_model(model, val, train, test)
    
    graph = plt.figure()
    
    #visualizing loss
    plt.plot(hist.history['loss'],color='blue',label='loss')
    plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
    graph.suptitle('Loss', fontsize=20)
    plt.legend(loc = "upper left")
    
    # Get date and time
    date_str = datetime.now().strftime("%Y-%m-%d")  # e.g., "2026-03-10"
    time_str = datetime.now().strftime("%H-%M-%S")   # e.g., "20-45-30"

    name = f"loss_plot_{date_str}_{time_str}.png"
    plt.savefig(name)
    
if __name__ == '__main__':
    main()