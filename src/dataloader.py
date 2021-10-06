import numpy as np
from tensorflow.keras.utils import Sequence
import os
import cv2

class DataLoader(Sequence):
    def __init__(self, file_path, batch_size=8, x_shape=(256, 256, 3), y_shape=(10,)):
        self.batch_size = batch_size
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.im_paths = os.listdir(file_path)
        self.indexes = np.arange(len(self.im_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.im_paths) / self.batch_size))

    def __getitem__(self, index):
        x_batch = np.empty((self.batch_size, *self.x_shape))
        y_batch = np.empty((self.batch_size, *self.y_shape))

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        for i in range(self.batch_size):
            image = self.im_paths[indexes[i]]
            x_batch[i,] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            y_batch[i,] = image

        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)