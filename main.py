import argparse

# import dataset


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.figure import Figure

import time
import numpy as np

import model
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
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0) #shuffle=False
    
    # #valid
    valid_list, valid_label = valid_data()
    valid_set = get_data(data_list = valid_list, label = valid_label, transform = valid_transform())
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0) #shuffle=False
    
            
    train_model = model.get_model()
    train_model.to(args.device)
    # print(train_model)
    
    optimizer = optim.Adam([
            {'params': train_model.paramgroup01(), 'lr': 1e-6},
            {'params': train_model.paramgroup234(), 'lr': 1e-4},
            {'params': train_model.parameters_classifier(), 'lr': 1e-2},
        ])
    
    loss_train = []
    loss_valid = []
    # figure = plt_loss(int(len(train_label)/args.batch_size) + 1)
    
    count = 0
    train_model.train()
    for data, label in tqdm(train_loader):
        # count+=1
        # if count==20:
        #     print(loss_all)
        #     break
        optimizer.zero_grad()
        
        # print("\n", data.dtype)
        # print(label)
        # time.sleep(10)
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
        loss_train.append(loss.item())
        print('\n', round(loss.item(), 3))
        # figure.update(round(loss.item(), 3))
        loss.backward()
        
        optimizer.step()
    
    
    # train_model.eval()
    # for data, label in tqdm(valid_loader):
    #     optimizer.zero_grad()
    
    #     data = data.to(args.device)
    #    # print(data.dtype)
    #     label.to(args.device)
    #     label = torch.unsqueeze(label, 1)
    #     label = label.type(torch.cuda.FloatTensor)
      
    #     outputs = train_model(data)
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
        self.x = []
        self.y = []
        self.batchs = batchs
        self.index = 0
        
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(7, 6))
        self.line1, = self.ax.plot(self.x, self.y)
        # plt.xlim(0, 20)
        # plt.ylim(0, 0.0001)
        plt.title('loss', fontsize=18)
        plt.xlabel("batchs", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        
        # plt.draw
    
    def update(self, loss):
        self.index += 1
        self.x.append(self.index)
        self.y.append(loss)
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(self.y)
        plt.autoscale(enable=True, axis='y') 
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        
if __name__ == '__main__':  
    args = parse()
    print(args)
    # plt_loss()
    # x = list(range(5))
    # y = [0, 1, 2 ,3 ,4]
    # for i in range(6, 10, 1):
        # x.append(i)
    loss_train = train(args)#, loss_valid = train(args)
    
    figure, ax = plt.subplots(figsize=(7, 6))
    line1, = ax.plot(list(range(len(loss_train))), loss_train)
    plt.title('loss', fontsize=18)
    plt.xlabel("batchs", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.draw
    #動態loss
    #統計pred正確與否
    #加入epoch -> valid ->混淆矩陣 ->儲存model 
    
    
