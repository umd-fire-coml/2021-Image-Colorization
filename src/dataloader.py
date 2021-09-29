import os
from torch.utils.data import DataLoader

for subdir, dirs, files in os.walk(r'./Places2 Dataset/Train'):
    training_data = files
for subdir, dirs, files in os.walk(r'./Places2 Dataset/Test'):
    testing_data = files
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
for batch in train_dataloader:
    print(batch.shape())
