import pytest
from os.path import exists

def test_downloader():
    assert(exists("model.py"))