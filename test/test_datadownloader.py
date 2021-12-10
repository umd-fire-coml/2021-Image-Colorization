import pytest
import os
from src.datadownloader import *


def test_downloader():
  url = 'http://file.fyicenter.com/a/sample.tar'
  save_path = './Dataset'
  dataset_name = 'test'
  download_url(url, dataset_name, save_path)
  goal_dir = os.path.join(os.getcwd(), f"{save_path}/{dataset_name}")
  assert(os.path.exists(goal_dir))
