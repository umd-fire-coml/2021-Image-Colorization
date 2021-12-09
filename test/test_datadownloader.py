import pytest
import os
from src.datadownloader import *


def test_downloader():
  url = 'http://ipv4.download.thinkbroadband.com/5MB.zip'
  save_path = '../Dataset'
  dataset_name = 'test'
  download_url(url, dataset_name)
  assert(len(os.listdir(save_path)))
