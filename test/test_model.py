import pytest
from tensorflow import keras
from src.dataloader import DataLoader
from src.model import Model


@pytest.fixture
def my_model():
    training = DataLoader('./data')
    validation = DataLoader('./data')
    model = Model(training=training, validation=validation)
    assert(isinstance(model.model, keras.Model))
    assert(isinstance(model.training_loader, DataLoader))
    assert(isinstance(model.validation_loader, DataLoader))
