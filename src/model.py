import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class Model(Sequential):
    def __init__(self, training, validation):
        self.my_model()
        self.training_loader = training
        self.validation_loader = validation

    # Create a model for training
    def my_model(self):
        self.add(Dense(5, activation='relu', input_shape=(4,)))
        self.add(Dense(10, activation='relu'))
        self.add(Dense(3, activation='softmax'))

    # Train the model
    def train_model(self):
        self.compile(
            optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        history = self.fit(
            self.training_loader, validation_data=self.validation_generator)
