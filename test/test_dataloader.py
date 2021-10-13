import pytest
from src.dataloader import DataLoader
import numpy

def test_dataloader():
    data_loader = DataLoader('./data')
    assert(data_loader.batch_size == 8)
    assert(data_loader.x_shape == (256, 256, 3))
    assert(data_loader.y_shape == (256, 256, 3))
    assert(isinstance(data_loader.im_paths, list[str]))
    assert(isinstance(data_loader.indexes, numpy.array))