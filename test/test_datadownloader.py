import pytest
import os
from src.datadownloader import *


def test_downloader():
  url = 'http://ipv4.download.thinkbroadband.com/5MB.zip'
  save_path = './data'
  download_url(url, save_path)
  assert(len(os.listdir(save_path)))
