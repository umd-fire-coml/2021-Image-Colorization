import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AvgPool2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, schedules
import datetime
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt

class ColorizationModel(Sequential):
    def __init__(self, train_data, val_data):
        super().__init__()
        self.my_model()
        self.train_data = train_data
        self.val_data = val_data

    # Create the layers of the model
    def my_model(self):
        # The tensor dimensions are tracked on the right
        self.add(tf.keras.Input(shape=(256,256,1))) #256,256,1
        self.add(Conv2D(4, 3, padding='same', activation='relu')) # 256,256,4
        self.add(AvgPool2D(2)) # 128,128,4
        self.add(Conv2D(16, 3, padding='same', activation='relu')) # 128,128,16
        self.add(AvgPool2D(2)) # 64,64,16
        self.add(Conv2D(64, 3, padding='same', activation='relu')) # 64,64,64
        self.add(AvgPool2D(2)) # 32,32,64
        self.add(Conv2D(128, 3, padding='same', activation='relu')) # 32,32,128
        self.add(AvgPool2D(2)) # 16,16,128
        self.add(Conv2D(256, 3, padding='same', activation='relu')) # 16,16,256
        self.add(UpSampling2D(2, interpolation='bilinear')) # 32,32,256
        self.add(Conv2D(128, 3, padding='same', activation='relu')) # 32,32,128
        self.add(UpSampling2D(2, interpolation='bilinear')) # 64,64,128
        self.add(Conv2D(64, 3, padding='same', activation='relu')) # 64,64,64
        self.add(UpSampling2D(2, interpolation='bilinear')) # 128,128,64
        self.add(Conv2D(16, 3, padding='same', activation='relu')) # 128,128,16
        self.add(UpSampling2D(2, interpolation='bilinear')) # 256,256,16
        self.add(Conv2D(4, 3, padding='same', activation='relu')) # 256,256,4
        self.add(Conv2D(2, 3, padding='same', activation='relu')) # 256,256,2
        # Create a learning rate scheduler
        lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.001,
                                                 decay_steps=10000,
                                                 decay_rate=0.9)
        # Compile the layers of the model using Adam optimizer, and MAE
        self.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='mean_absolute_error',
            metrics=['mean_absolute_error'])

    # Train the model
    def train_model(self,
                    epochs=100,
                    initial_epoch=0,
                    weights_path='../checkpoints/weights.hdf5',
                    best_weights_path='../checkpoints/bestweights.hdf5'):
        # If the weights_path exists and we are resuming previous training, 
        # load the stored weights into the model.
        if os.path.exists(weights_path) and initial_epoch:
            print("Loading previously saved weights.")
            self.load_weights(weights_path)
        if initial_epoch:
            print(f"Resuming training from epoch {initial_epoch}.")
        # Create the checkpoint callback, monitors for the minimum loss on the 
        # validation set.
        best_ckpt_cb = ModelCheckpoint(best_weights_path,
                                  monitor='val_loss',
                                  save_best_only=True,
                                  mode='min',
                                  save_weights_only=True)
        ckpt_cb = ModelCheckpoint(weights_path,
                                  save_best_only=False,
                                  save_weights_only=True)
        # Create the Tensorboard callback
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # Create callback list
        callbacks = [best_ckpt_cb, ckpt_cb, tb_cb]
        # Train the model starting at initial_epoch until epochs+initial_epoch.
        self.fit(x=self.train_data,
                 epochs=epochs+initial_epoch,
                 callbacks=callbacks,
                 validation_data=self.val_data,
                 initial_epoch=initial_epoch
                )
    
    # Test a given image with the model
    def test_image(self, path, save_path):
        # Open the image
        rgb = tf.image.decode_jpeg(tf.io.read_file(path), 3)
        # Convert to YCbCr
        ycbcr = tfio.experimental.color.rgb_to_ycbcr(rgb)
        # Convert back to RGB for visualization
        bgr_back = cv2.cvtColor(ycbcr.numpy(), cv2.COLOR_YCrCb2BGR)
        # Normalize to [0,1]
        ycbcr_float32 = tf.cast(ycbcr, tf.float32) / 255.0
        # Shape the Y channel as a batch
        test_batch = np.reshape(ycbcr_float32[:,:,0], (1, 256, 256, 1))
        # Get the resulting channels
        result = self(test_batch)[0]
        # Stack the channels to get a result YCbCr
        ycbcr_result_float32 = np.dstack((ycbcr_float32[:,:,0], result))
        # Multiply by 255 to get [0,255] YCbCr Image
        ycbcr_result = tf.cast(ycbcr_result_float32 * 255.0, tf.uint8)
        # Convert the YCbCr result to RGB
        bgr_result = cv2.cvtColor(ycbcr_result.numpy(), cv2.COLOR_YCrCb2BGR)
        # Save the resulting image
        cv2.imwrite(save_path, bgr_result)
        # Show the resulting image with grayscale and real images
        f, axarr = plt.subplots(1, 3, figsize=(21,21))
        axarr[0].imshow(ycbcr[:,:,0], cmap='gray')
        axarr[0].set_title("Grayscale")
        axarr[0].axis('off')
        axarr[1].imshow(bgr_result)
        axarr[1].set_title("Result")
        axarr[1].axis('off')
        axarr[2].imshow(bgr_back)
        axarr[2].set_title("Real")
        axarr[2].axis('off')
        plt.show()
        # Display the real and predicted Cb and Cr channels
        f, axarr = plt.subplots(2, 2, figsize=(14,14))
        axarr[0][0].imshow(ycbcr_result[:,:,1], cmap='gray')
        axarr[0][0].set_title("Cb Result")
        axarr[0][0].axis('off')
        axarr[0][1].imshow(ycbcr[:,:,1], cmap='gray')
        axarr[0][1].set_title("Cb Real")
        axarr[0][1].axis('off')
        axarr[1][0].imshow(ycbcr_result[:,:,2], cmap='gray')
        axarr[1][0].set_title("Cr Result")
        axarr[1][0].axis('off')
        axarr[1][1].imshow(ycbcr[:,:,2], cmap='gray')
        axarr[1][1].set_title("Cr Real")
        axarr[1][1].axis('off')
        plt.show()
