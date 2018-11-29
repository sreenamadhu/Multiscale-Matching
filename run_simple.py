
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import pandas as pd
from skimage import io
import os
import numpy as np
from torchvision import datasets, transforms
import torch
import argparse
from CrossCNN_simple import *
from random import randint
class PairDataset(data.Dataset):

    def __init__(self, list,split_str = '',
                transform=None, mirror=False):

        self.pairs = pd.read_csv(list, sep = ' ', header = None)
        self.transform = transform
        self.mirror = mirror
        self.list = list


    def __getitem__(self, index):
        
        line = self.pairs.ix[index,0]
        line = line.split(',')

        im1 = io.imread('/home/sreena/Desktop/Ran/dataset/' + line[0])
        im2 = io.imread('/home/sreena/Desktop/Ran/dataset/' + line[1])
        label = int(line[2])



        im1 = Image.fromarray(im1, mode='RGB')
        im2 = Image.fromarray(im2, mode='RGB')

        ind1 = randint(0,4)
        ind2 = randint(0,4)

        if self.transform is not None:

            if len(self.transform) > 1:

                im1 = self.transform[ind1](im1)
                im2 = self.transform[ind2](im2)

            else:
                im1 = self.transform[0](im1)
                im2 = self.transform[0](im2)

        return im1,im2,label

    def __len__(self):
        return len(self.pairs)


parser = argparse.ArgumentParser(description='Cross CNN Example')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--modelname', type=str, default='Cross-CNN_')
parser.add_argument('--pretrained', type=bool, default=True)

args = parser.parse_args()

rotation = transforms.Compose([transforms.RandomRotation(10),
                            transforms.ToTensor()])
h_crop = transforms.Compose([transforms.Resize((153+5,116)),
                            transforms.RandomCrop((153,116)),
                            transforms.ToTensor()])
w_crop = transforms.Compose([transforms.Resize((153,116+5)),
                            transforms.RandomCrop((153,116)),
                            transforms.ToTensor()])
crop = transforms.Compose([transforms.Resize((153+15,116+15)),
                            transforms.RandomCrop((153,116)),
                            transforms.ToTensor()])
normal = transforms.Compose([transforms.ToTensor()])


train_loader = torch.utils.data.DataLoader(
    PairDataset('/home/sreena/Desktop/Ran/dataset/pair_train_crop_new.txt', '',
                transform= [rotation,h_crop,w_crop,crop,normal]),
    batch_size=args.batch_size)

valid_loader = torch.utils.data.DataLoader(
    PairDataset('/home/sreena/Desktop/Ran/dataset/pair_valid_crop_new.txt', '',
                transform=[normal]),
    batch_size=args.batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    PairDataset('/home/sreena/Desktop/Ran/dataset/pair_test_crop_new.txt', '',
                transform=[normal]),
    batch_size=args.batch_size, shuffle=False, num_workers=2)

model = CrossCNN()
model.pair_fit(train_loader, valid_loader, lr=args.lr, num_epochs=args.epochs,train_log = 'logs/pairwise_simple.txt')

