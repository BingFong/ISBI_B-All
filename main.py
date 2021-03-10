import argparse

# import dataset
import model

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import train_data, get_data, train_transform, valid_data, valid_transform

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--out', default='results', help='output folder')
    return parser.parse_args()


def train(args):

    train_list, train_label = train_data()
    train_set = get_data(data_list = train_list, label = train_label, transform = train_transform())
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=False, num_workers=0) #shuffle=False
    
    # #valid
    # valid_list, valid_label = valid_data()
    # valid_set = get_data(data_list = valid_list, label = valid_label, transform = valid_transform())
    # valid_loader = DataLoader(
    #     dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0) #shuffle=False
    
            
    train_model = model.get_model()
    train_model.to(args.device)
    # print(train_model)
    
    optimizer = optim.Adam([
            {'params': train_model.paramgroup01(), 'lr': 1e-6},
            {'params': train_model.paramgroup234(), 'lr': 1e-4},
            {'params': train_model.parameters_classifier(), 'lr': 1e-2},
        ])
    
    for data, label in tqdm(train_loader):
        # print("\n", data.dtype)
        data = data.to(args.device)
        # print(data.dtype)
        label.to(args.device)
        label = torch.unsqueeze(label, 1)
        label = label.type(torch.cuda.FloatTensor)
        
        outputs = train_model(data)
        # print('size:', outputs.size())
        # print('val:', outputs)
        
        loss = torch.nn.BCEWithLogitsLoss()
        loss = loss(outputs, label)
        loss.backward()
        optimizer.step()
        
    print('Finished Training')   
    
if __name__ == '__main__':  
    args = parse()
    print(args)
    
    train(args)
    
    
    #動態loss
    #統計pred正確與否
    #加入epoch -> valid ->混淆矩陣 ->儲存model 
    
    
