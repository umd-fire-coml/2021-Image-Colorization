import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.utils import Sequence
import glob

class DataLoader(Sequence):
    def __init__(self, file_path, batch_size=128):
        self.batch_size = batch_size
        self.x_shape = (256, 256)
        self.y_shape = (256, 256, 2)
        # Get all recursive image paths
        self.im_paths = glob.glob(file_path + '/**/*.jpg', recursive=True)
        self.indexes = np.arange(len(self.im_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Create empty output arrays
        x_batch = np.empty((self.batch_size, *self.x_shape))
        y_batch = np.empty((self.batch_size, *self.y_shape))
        # Calculate the Indexes we want to access
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # For each image, x_batch is the GRAYSCALE and y_batch is the original image
        # Normalize each image to the range [0, 1]
        for i in range(self.batch_size):
            image = tf.image.decode_jpeg(tf.io.read_file(self.im_paths[indexes[i]]), 3)
            ycbcr = tfio.experimental.color.rgb_to_ycbcr(image)
            ycbcr_float32 = tf.cast(ycbcr, tf.float32) / 255.0
            y_float32 = ycbcr_float32[:,:,0]
            cbcr_float32 = ycbcr_float32[:,:,1:]
            x_batch[i,] = y_float32
            y_batch[i,] = cbcr_float32
        return x_batch, y_batch

    def on_epoch_end(self):
        # Shuffle the order in which the images are obtained
        np.random.shuffle(self.indexes)
