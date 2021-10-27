import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class Model(object):
    def __init__(self, training, validation):
        self.model = self.my_model()
        self.training_loader = training
        self.validation_loader = validation

    # Create a model for training
    def my_model(self):
        model = Sequential()

        model.add(Dense(5, activation='relu', input_shape=(4,)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        return model

    # Train the model
    def train_model(self):
        self.model.compile(
            optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit(
            self.training_loader, validation_data=self.validation_generator)
