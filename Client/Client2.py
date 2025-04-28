import os
import csv
import flwr as fl
import tensorflow as tf
import time
from model import build_model

# Log
log_path = "#change to path that where you want yout log file to be saved" 
log_file = os.path.join(log_path, "Log_Client1.csv")
if not os.path.exists(log_path):
    os.makedirs(log_path)

batch_size = 16
img_size = (160, 160)

# Load data
data_dir = "#change to path of your dataset file" 
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory {data_dir} not found.")

def load_data():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )
    return train_data, val_data

#Flower Initiation
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_model()
        self.train_data, self.val_data = load_data()
        self.current_round = 0

        # Log's file initiation
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Fit Duration (s)"])

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        start_time = time.time()
        
        history = self.model.fit(
            self.train_data,
            epochs=10,
            validation_data=self.val_data,
            verbose=0
        )
        
        end_time = time.time()
        duration = end_time - start_time

        self.current_round += 1
        train_loss = history.history["loss"][0]
        train_acc = history.history["accuracy"][0]
        val_loss = history.history["val_loss"][0]
        val_acc = history.history["val_accuracy"][0]

        # Save log's file
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_round, train_loss, train_acc, val_loss, val_acc, round(duration, 2)])
        return self.model.get_weights(), len(self.train_data), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.val_data, verbose=0)
        return loss, len(self.val_data), {"accuracy": acc}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="xxx.xxx.xxx.xxx:xxxx", client=FLClient()) #Fill x with your server IP Addresses and port
