from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import os
import glob
from pathlib import Path
import pandas as pd

class get_data(Dataset):
    def __init__(self, train_list, label, transform):
        self.train_list = train_list
        self.label = label
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(train_list[index])

        if self.transform is not None:
            img = self.transform(img)
        # img = img.unsqueeze(0)
        return img, label[index]
        
    def __len__(self):
        return len(train_list)

def train_transform():
    return transforms.Compose([
        transforms.CenterCrop(300),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=360, translate=(0.2, 0.2)),
        transforms.Resize((224,224), Image.LANCZOS),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0, 0, 0],
                             # std=[1, 1, 1])
    ])

def valid_transform():
    return transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((224,224), Image.LANCZOS),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0, 0, 0],
        #                      std=[1, 1, 1])
    ])

def train_data():
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_root = r'data'
    train_list = glob.glob(os.path.join(current_path, data_root, '*', '*', '*.bmp'))
    
    label = []
    for data in train_list:
       data = Path(data)
       parts = data.parts
       if parts[5] == 'all':
           label.append(1)
       else:
           label.append(0)
    
    return train_list, label

def valid_data():
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_root = r'data'
    valid_list = glob.glob(os.path.join(current_path, data_root, '*', '*.bmp'))
    phase2 = os.path.join(current_path, data_root, 'phase2.csv')
    
    
    df = pd.read_csv(phase2)
    label = df['labels'].values.tolist()
    
    return valid_list, label
   
if __name__ == '__main__':
    train_list, train_label = train_data()
    valid_list, valid_label = valid_data()
    
    train_set = get_data(train_list = train_list, label = train_label, transform = train_transform())
    
    train_loader = DataLoader(
        dataset=train_set, batch_size=10, shuffle=False, num_workers=0) #shuffle=False
    
    
    
    # for data in train_loader:
    #     print('Size of image:', data[0].shape)  # batch_size * 3 * 300 * 300
        
