import pytest
import os
from src.datadownloader import *


def test_downloader():
  url = 'https://www.learningcontainer.com/download/sample-tar-file-download-for-testing/?wpdmdl=2487&refresh=61754a613701e1635076705'
  save_path = './data'
  dataset_name = 'test'
  download_url(url, dataset_name, save_path)
  assert(len(os.listdir(save_path)))
