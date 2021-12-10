import pytest
from src.dataloader import DataLoader
import os

def test_dataloader():
    goal_dir = os.path.join(os.getcwd(), "test")
    data_loader = DataLoader(goal_dir)
    assert(data_loader.batch_size == 128)
    assert(data_loader.x_shape == (256, 256))
    assert(data_loader.y_shape == (256, 256, 2))
    assert(isinstance(data_loader.im_paths, list))
    assert(isinstance(data_loader.im_paths, list))