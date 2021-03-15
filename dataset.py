from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import os
import glob
from pathlib import Path
import pandas as pd

class get_data(Dataset):
    
    def __init__(self, data_list, label, transform):
        self.data_list = data_list
        self.label = label
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(self.data_list[index])

        if self.transform is not None:
            img = self.transform(img)
            
        return img, self.label[index]
        
    def __len__(self):
        return len(self.data_list)

def train_transform():
    #compose training transformation
    return transforms.Compose([
        transforms.CenterCrop(300),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=360, translate=(0.2, 0.2)),
        transforms.Resize((224,224), Image.LANCZOS),
        transforms.ToTensor(),
    ])

def valid_transform():
    #compose validaing transformation
    return transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((224,224), Image.LANCZOS),
        transforms.ToTensor(),
    ])

def train_data():
    #return training set directory list and labels
    
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_root = r'data'
    train_list = glob.glob(os.path.join(
        current_path, data_root, 'fold_0', '*', '*.bmp'))
    
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
    #return validating set directory list and labels
    
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_root = r'data'
    valid_list = glob.glob(os.path.join(current_path, data_root, 'phase2', '*.bmp'))
    phase2 = os.path.join(current_path, data_root, 'phase2.csv')
    
    
    df = pd.read_csv(phase2)
    label = df['labels'].values.tolist()
    
    return valid_list, label
    
if __name__ == '__main__':
    
    train_list, train_label = train_data()
    train_set = get_data(data_list = train_list, label = train_label, transform = train_transform())
    train_loader = DataLoader(
        dataset=train_set, batch_size=10, shuffle=True, num_workers=0) #shuffle=False
    
    valid_list, valid_label = valid_data()
    valid_set = get_data(data_list = valid_list, label = valid_label, transform = valid_transform())
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=10, shuffle=False, num_workers=0) #shuffle=False
    
    # for i, data in enumerate(train_loader, 0):
    #     print(i)
    #     print(data[1].shape)
        
        
