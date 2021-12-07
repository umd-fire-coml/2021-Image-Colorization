import os
import requests
import tarfile

def download_url(url, dataset_name, save_path='../Dataset', chunk_size=2048):
    # If the dataset is already downloaded, skip to untaring and return
    if os.path.exists(f"{save_path}/{dataset_name}.tar"):
        print(f"{dataset_name}.tar already exists, skipping downloading.")
        untar(dataset_name, save_path)
        return
    # Otherwise,
    print(f"Downloading {dataset_name}.tar")
    # Make the save_path folder if it doesn't exist
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass
    # Request the URL
    r = requests.get(url, stream=True)
    # Open the tarpath in create write binary mode
    with open(f"{save_path}/{dataset_name}.tar", 'wb') as tarobj:
        # Chunk the response and write each chunk
        for chunk in r.iter_content(chunk_size=chunk_size):
            tarobj.write(chunk)
    print(f"{dataset_name}.tar successfully downloaded.")
    untar(dataset_name, save_path)

def untar(dataset_name, save_path='../Dataset'):
    # If the dataset is already untarred, return
    if os.path.exists(f"{save_path}/{dataset_name}"):
        print(f"{dataset_name}.tar already untarred, skipping untarring.")
        return
    # Otherwise,
    print(f"Untarring {dataset_name}.tar")
    # With the tarfile opened, extract the entire tarfile
    with tarfile.open(f"{save_path}/{dataset_name}.tar") as tar:
        tar.extractall(f"{save_path}/{dataset_name}")
    print(f"{dataset_name}.tar successfully untarred.")