import pytest
from dataloader import DataLoader

def test_dataloader():
    data_loader = DataLoader()
    assert(data_loader.test_dataloader is not None)
    assert(data_loader.train_dataloader is not None)