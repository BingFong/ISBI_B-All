import model as Model
from dataset import train_data, get_data, train_transform, valid_data, valid_transform

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR


class plt_loss():
    #建立畫布，x軸值設為總batch數
    def __init__(self, batchs, mode):
        self.x = [0]
        self.y = [0]
        self.batchs = batchs
        
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(7, 6))
        self.line1, = self.ax.plot(self.x, self.y)
        plt.xlim(0, batchs)
        plt.ylim(0, 4)
        plt.title(mode, fontsize=18)
        plt.xlabel("batchs", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        
    def update(self, loss):
        #更新並重繪
        self.y.append(loss)
        self.line1.set_xdata(list(range(len(self.y))))
        self.line1.set_ydata(self.y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
def saving_loss_and_figure(loss_batchs, loss_amount, mode):
    #saving averageg loss depands on which mode
    folder = os.path.join(args.folder, mode)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    averaged_loss = sum(loss_batchs) / loss_amount
    print('averaged {0} loss: {1}'.format(mode, averaged_loss))
    dict = {'averaged_train_loss': [averaged_loss]}
    df = pd.DataFrame(dict)
    df.to_csv(folder + '/' + mode + '.csv') #saving loss as csv
    
    plt.savefig(folder + '/' + 'loss') #saving loss plot
    plt.close()
    
def train(model, train_list, train_label, train_loader, optimizer):
    print('\nstart training')
    model.train()
    batchs = int(len(train_label)/args.batch_size) + 1
    figure = plt_loss(batchs, 'training') #建立畫布
    
    loss_train = []
    pred_train = np.empty([1, 1])
    pred_label = np.empty([1, 1])
    
    
    for data, label in tqdm(train_loader):
        #traing flow
        data = data.to(args.device)
        label.to(args.device)
        label = torch.unsqueeze(label, 1)
        label = label.type(torch.cuda.FloatTensor)
        
        outputs = model(data)
        
        #pred 
        pred = torch.sigmoid(outputs.data).cpu().numpy()
        pred = np.where(pred>0.5 , 1, 0)
        pred_train = np.concatenate((pred_train, pred), axis=0)
        pred_label = np.concatenate((pred_label, label.cpu()), axis=0)
        
        #loss
        pos_weight=torch.FloatTensor([0.3156/0.6844]).cuda().expand(len(label))
        pos_weight = torch.reshape(pos_weight, (len(label), 1))
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss(outputs, label)
        loss_train.append(loss.item() * list(label.size())[0])
        figure.update(round(loss.item(), 7)) #dynamic drawing
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    pred_train = np.delete(pred_train, 0, axis=0)
    pred_train.astype(int)
    pred_label = np.delete(pred_label, 0, axis=0)
    pred_label.astype(int)
    
    # pred_label = np.asarray(train_label)
    # pred_label  = np.reshape(pred_label , (len(pred_label ), 1))
    acc = accuracy_score(pred_train, pred_label)
    print('train acc:', acc)
    
    saving_loss_and_figure(loss_train, len(train_label), 'train')
    
    print('Finished Training\n')
    time.sleep(3)
 
def valid(model, valid_list, valid_label, valid_loader, optimizer, e):
    print('\nstart valid')
    model.eval()
    batchs = int(len(valid_label)/args.batch_size) + 1
    figure = plt_loss(batchs, 'validation') #建立畫布
    
    loss_valid = []
    pred_valid = np.empty([1, 1])
    
    for data, label in tqdm(valid_loader):
        #validation flow
        
        data = data.to(args.device)
        label.to(args.device)
        label = torch.unsqueeze(label, 1)
        label = label.type(torch.cuda.FloatTensor)
        
        outputs = model(data)
        
        #pred 
        pred = torch.sigmoid(outputs.data).cpu().numpy()
        
        pred = np.where(pred>0.5 , 1, 0)
        pred_valid = np.concatenate((pred_valid, pred), axis=0)
      
        #loss
        pos_weight=torch.FloatTensor([0.3156/0.6844]).cuda().expand(len(label))
        pos_weight = torch.reshape(pos_weight, (len(label), 1))
        loss = torch.nn.BCEWithLogitsLoss(pos_weight =pos_weight)
        loss = loss(outputs, label)
        loss_valid.append(loss.item())
        figure.update(loss.item())
        
    pred_valid = np.delete(pred_valid, 0, axis=0)
    pred_valid.astype(int)
    pred_label = np.asarray(valid_label)
    pred_label  = np.reshape(pred_label , (len(pred_label ), 1))
    acc = accuracy_score(pred_valid, pred_label )
    print('valid_acc:', acc)
    
    saving_loss_and_figure(loss_valid, len(valid_label), 'validation') 
    
    df = pd.DataFrame(pred_valid, columns=["colummn"])
    df.to_csv(args.folder + '/' + str(e) +'_pred_valid.csv', index=False)
    
    print('Finished valid')
    time.sleep(3)
    
def schedule(epoch):
    if epoch <4 :
        ub = 1 - (epoch * 0.01)
    else:
        ub = 0.007
    return ub

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
    
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda e: schedule(e),
                                         lambda e: schedule(e),
                                         lambda e: schedule(e)])
    for e in range(3):
        print('epoch: {0}'.format(e))
        train(model, train_list, train_label, train_loader, optimizer)
        scheduler.step()
        
        valid(model, valid_list, valid_label, valid_loader, optimizer, e)
    
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
    
    train_valid(args)
    
    #加入epoch -> valid ->混淆矩陣 ->儲存model 
