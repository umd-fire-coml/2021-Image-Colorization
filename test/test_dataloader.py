import pytest
from src.dataloader import DataLoader

def test_dataloader():
    assert(DataLoader is not None)