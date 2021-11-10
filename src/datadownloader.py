import os
import requests

def download_url(url, save_path, chunk_size=128):
  try:
    os.mkdir(save_path)
  except FileExistsError:
    pass
  r = requests.get(url, stream=True)
  with open(f"{save_path}/data", 'wb') as fd:
    for chunk in r.iter_content(chunk_size=chunk_size):
      fd.write(chunk)