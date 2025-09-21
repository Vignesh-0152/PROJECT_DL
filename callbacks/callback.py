import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    TerminateOnNaN,
    CSVLogger
)
import os

class CustomCallback:
    def __init__(self, 
                patience=5, 
                best_path=          f'trained_model/model/best_model.keras',
                last_path =         f'trained_model/model/last_model.keras',
                model_log_path =    f'trained_model/logs/tensor_board',
                csv_logger =        f'trained_model/logs/csv_logger.csv'
            ):
        self.patience =         patience
        self.best_path =        best_path
        self.last_path =        last_path
        self.model_log_path =   model_log_path
        self.csv_logger =       csv_logger

        os.makedirs('trained_model/model', exist_ok=True)
        os.makedirs('trained_model/logs', exist_ok=True)

        self.Callback_list = self.get_callback()

    def get_callback(self):

        #Early Stopping
        EarlyStop = EarlyStopping(
            monitor='val_loss', 
            patience=self.patience, 
            restore_best_weights=True
        )

        #Model check point for the best model
        Best_model = ModelCheckpoint(
            filepath=self.best_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

        #Model chekc point for the last model
        Last_model = ModelCheckpoint(
            filepath = self.last_path,
            monitor = 'val_loss',
            save_best_only= False,
            save_weights_only= False,
            mode= 'min',
            verbose=0
        )

        #Reduce on plateau
        reduce_on_plateau = ReduceLROnPlateau(
            monitor= 'val_loss',
            factor= 0.1,
            patience= 3,
            verbose= 1,
            mode='min',
            cooldown= 3,
            min_lr= 1e-6
        )

        #tensorboard
        Tensor_board_log = TensorBoard(
            log_dir= self.model_log_path,
            histogram_freq= 1,
            write_graph= True,
            write_images= False,
            update_freq= 'epoch',
            profile_batch= 0
        )

        #CSV Logger
        scv_logger = CSVLogger(
            filename= self.csv_logger,
            append= True
        )

        #Terminate on NaN
        terminate_on_nan = TerminateOnNaN()

        return [
            EarlyStop,
            Best_model,
            Last_model,
            reduce_on_plateau,
            Tensor_board_log,
            scv_logger,
            terminate_on_nan
        ]