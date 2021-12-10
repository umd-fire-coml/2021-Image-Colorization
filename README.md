# 2021 Image-Colorization #
## Product Description
This image colorization model takes an input image, convert it to greyscale, then creates a realistic colorization of the image based off of the trained model. The model architecture is based off a YCbCr channel model in order to properly colorize the image, as well as using convolutional layers to build the model. The model was trained using the places365 dataset as ground truth images for the colorization, and a greyscale version of the places365 dataset as training images. Our model is based off of the model used 
in "Colorful Image Colorization" by Richard Zhang et al. [1]. 

## Video Demonstration
[[Video Demonstration](https://www.youtube.com/watch?v=F8dwnLsyd0s)]

## Colab File 
https://github.com/umd-fire-coml/2021-Image-Colorization/blob/final-deliverable/src/main.ipynb


## Directory Guide

- .github/workflows/run_all_tests.yaml: A file that downloads the required libraries for the tests and runs each one.
- checkpoints/bestweights.hdf5: The most accurate saved weights from training the model.
- checkpoints/weights.hdf5: The most recent saved weights from training the model.
- src/datadownloader.py: Contains methods that download a target URL as a .tar file, then unzips the .tar file to a destination.
- src/dataloader.py: Resizes the images to a standard size then loads the images from the dataset into two arrays, one for grayscale and one for the original image.
- src/model.py: Builds the model, trains the model using the grayscale and ground truth images, and displays the resulting images. 
- src/main.ipynb: A colab file which demonstrates the environment setup, building and training of the model, and the resulting images.
- test/test_datadownloader.py: A test for the datadownloader.py functions.
- test/test_dataloader.py: Tests for the dataloader.py functions.
- test/test_model.py: Tests for the model.py functions. 
- requirements.txt: The requirements needed to run the program.
- test-requirements.txt: The requirements needed to run the test files.
- setup.sh: Installs all requirements needed for running and testing the program using pip install.


## Environment Setup
If you are not using Google Colab, you will need to run the following:
```bash
setup.sh
```
This will generate the environment and install all requirements for the program and tests.

## Dataset Download
In order to download the dataset, you want to run the following:
```bash
download_url('http://data.csail.mit.edu/places/places365/test_256.tar', 'train')
download_url('http://data.csail.mit.edu/places/places365/val_256.tar', 'val')
```
This will download and unzip the places365 dataset, setting one instance as training images and the other as validation images.

## Training and testing the model
Steps about training and testing the model are included in our main.ipynb file, which can be found here: 
https://github.com/umd-fire-coml/2021-Image-Colorization/blob/final-deliverable/src/main.ipynb

## Citations
[1] R. Zhang, P. Isola, and A. A. Efros, “Richzhang/colorization: Automatic colorization using deep neural networks. ‘Colorful image colorization." in ECCV, 2016.,” GitHub, 12-Oct-2016. [Online]. Available: https://github.com/richzhang/colorization. [Accessed: 10-Dec-2021]. 
