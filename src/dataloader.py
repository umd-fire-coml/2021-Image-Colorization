import os
from torch.utils.data import DataLoader

class DataLoader():
    def __init__(self, train_path = r'./Places2 Dataset/Train', test_path = r'./Places2 Dataset/Test'):
        for subdir, dirs, files in os.walk(train_path):
            training_data = files
        for subdir, dirs, files in os.walk(test_path):
            testing_data = files
        self.train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
        for batch in self.train_dataloader:
            print(batch.shape())
        for batch in self.train_dataloader:
            print(batch.shape())
