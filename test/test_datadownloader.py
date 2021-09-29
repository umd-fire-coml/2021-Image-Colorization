import pytest
from datadownloader import DataDownloader

def test_downloader():
  assert(DataDownloader.getPath() != "")
