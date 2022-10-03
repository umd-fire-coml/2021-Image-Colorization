import os
import requests
import tarfile

def download_url(url, dataset_name, save_path='./Dataset', chunk_size=2048):
    # If the dataset is already downloaded, skip to untaring and return
    make_dir = os.path.join(os.getcwd(), save_path)
    goal_dir = os.path.join(os.getcwd(), f"{save_path}/{dataset_name}.tar")
    if os.path.exists(goal_dir):
        print(f"{dataset_name}.tar already exists, skipping downloading.")
        untar(dataset_name, save_path)
        return
    # Otherwise,
    print(f"Downloading {dataset_name}.tar")
    # Make the save_path folder if it doesn't exist
    try:
        os.mkdir(make_dir)
    except FileExistsError:
        pass
    # Request the URL
    r = requests.get(url, stream=True)
    # Open the tarpath in create write binary mode
    with open(goal_dir, 'wb') as tarobj:
        # Chunk the response and write each chunk
        for chunk in r.iter_content(chunk_size=chunk_size):
            tarobj.write(chunk)
    print(f"{dataset_name}.tar successfully downloaded.")
    untar(dataset_name, save_path)

def untar(dataset_name, save_path='./Dataset'):
    # If the dataset is already untarred, return
    goal_dir = os.path.join(os.getcwd(), f"{save_path}/{dataset_name}")
    tar_path = os.path.join(os.getcwd(), f"{save_path}/{dataset_name}.tar")
    if os.path.exists(goal_dir):
        print(f"{dataset_name}.tar already untarred, skipping untarring.")
        return
    # Otherwise,
    print(f"Untarring {dataset_name}.tar")
    # With the tarfile opened, extract the entire tarfile
    with tarfile.open(tar_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner) 
            
        
        safe_extract(tar, goal_dir)
    print(f"{dataset_name}.tar successfully untarred.")