import model as Model
from dataset import train_data, get_data, train_transform, valid_data, valid_transform

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_valid(args):

    train_list, train_label = train_data()
    train_set = get_data(data_list = train_list, label = train_label, transform = train_transform())
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0) #shuffle=False
    
    valid_list, valid_label = valid_data()
    valid_set = get_data(data_list = valid_list, label = valid_label, transform = valid_transform())
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0) #shuffle=False
    
            
    model = Model.get_model()
    model.to(args.device)
    
    optimizer = optim.Adam([
            {'params': model.paramgroup01(), 'lr': 1e-6},
            {'params': model.paramgroup234(), 'lr': 1e-4},
            {'params': model.parameters_classifier(), 'lr': 1e-2},
        ])
    
    batchs = int(len(train_label)/args.batch_size) + 1
    figure = plt_loss(batchs) #建立畫布

    loss_train = []
    
    model.train()
    for data, label in tqdm(train_loader):
        #traing flow
        data = data.to(args.device)
        label.to(args.device)
        label = torch.unsqueeze(label, 1)
        label = label.type(torch.cuda.FloatTensor)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = torch.nn.BCEWithLogitsLoss()
        loss = loss(outputs, label)
        loss_train.append(loss.item())
        figure.update(round(loss.item(), 7)) #dynamic drawing
        loss.backward()
        
        optimizer.step()
    
    plt.savefig(os.path.join(args.folder, 'loss')) #saving loss plot
    
    # model.eval()
    # for data, label in tqdm(valid_loader):
    #     optimizer.zero_grad()
    
    #     data = data.to(args.device)
    #    # print(data.dtype)
    #     label.to(args.device)
    #     label = torch.unsqueeze(label, 1)
    #     label = label.type(torch.cuda.FloatTensor)
      
    #     outputs = model(data)
    #    # print('size:', outputs.size())
    #    # print('val:', outputs)
      
    #     loss = torch.nn.BCEWithLogitsLoss()
    #     loss = loss(outputs, label)
    #     loss_valid.append(loss.item())
       # print('\n', loss.item())
        # figure.update(loss.item())
        # loss.backward()
        
     #    optimizer.step()
    print('Finished Training')   
    return loss_train#, loss_valid
    
class plt_loss():
    def __init__(self, batchs):
        self.x = [0]
        self.y = [0]
        self.batchs = batchs
        
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(7, 6))
        self.line1, = self.ax.plot(self.x, self.y)
        plt.xlim(0, batchs)
        plt.ylim(0, 4)
        plt.title('loss', fontsize=18)
        plt.xlabel("batchs", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        
    def update(self, loss):
        self.y.append(loss)
        self.line1.set_xdata(list(range(len(self.y))))
        self.line1.set_ydata(self.y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        
if __name__ == '__main__':  
    
    def parse():
        #experimental parameters and directory settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--epochs', type=int, default=8)
        parser.add_argument('--out', default='results', help='output root folder')
        parser.add_argument('--folder', default=time.strftime("%Y%m%d_%H%M"), help='output folder')
        
        return parser.parse_args()

    args = parse()
    folder = os.path.join(args.out, args.folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    args.folder = folder
    print(args)
    
    loss_train = train_valid(args)
    
    
    #動態loss
    #統計pred正確與否
    #加入epoch -> valid ->混淆矩陣 ->儲存model 
    
    
