import pytest
from tensorflow.keras.models import Sequential
from src.dataloader import DataLoader
from src.model import ColorizationModel


@pytest.fixture
def my_model():
    training = DataLoader('./data')
    validation = DataLoader('./data')
    model = ColorizationModel(training, validation)
    assert(issubclass(model, Sequential))
    assert(isinstance(model.train_data, DataLoader))
    assert(isinstance(model.val_data, DataLoader))

