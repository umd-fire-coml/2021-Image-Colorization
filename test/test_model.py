import pytest
from tensorflow.keras.models import Sequential
from src.dataloader import DataLoader
from src.model import ColorizationModel
import os

def my_model():
    goal_dir = os.path.join(os.getcwd(), "test")
    data_loader = DataLoader(goal_dir)
    model = ColorizationModel(training=data_loader, validation=data_loader)
    assert(isinstance(model, Sequential))
    assert(isinstance(model.train_data, DataLoader))
    assert(isinstance(model.val_data, DataLoader))
