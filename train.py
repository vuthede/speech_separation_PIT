

#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import sys
from models.model import AudioOnlyModel
from torchvision import  transforms
import math
from datasets.VoiceMixtureDataSet import AudioMixtureDataset, loss_func2
from logger.TensorboardLogger import TensorBoardLogger
from tqdm import tqdm

DEBUG = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device -------------------------------------:", device)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])


transform_valid = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f'Save checkpoint to {filename}')




def train_one_epoch(traindataloader, model, optimizer, epoch, args=None, tensorboardLogger=None):
    model.train()
    losses= AverageMeter()
    num_batch = len(traindataloader)
    i = 0

    with tqdm(traindataloader, unit="batch") as tepoch:
        for (X, Y) in tepoch:
            
            i += 1
            # if i>=40:
            #     break
        
            tepoch.set_description(f"Epoch {epoch}")
            X = X.to(device)
            
            # Inference model to generate heatmap
            Y_pred = model(X)

            # Contrastive loss
            loss = loss_func2(S_true=Y,S_pred=Y_pred, gamma=0.1, num_speaker=2)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            if DEBUG:
                tensorboardLogger.log(f"train/loss", loss.item(), epoch*num_batch+i)

            
            tepoch.set_postfix(loss=loss.item())

        # Log train averge loss in each dataset
        tensorboardLogger.log(f"train/loss_avg", losses.avg, epoch)

     
    return losses.avg

def validate(valdataloader, model, optimizer, epoch, args, tensorboardLogger=None):
    if not os.path.isdir(args.snapshot):
        os.makedirs(args.snapshot)

    logFilepath  = os.path.join(args.snapshot, args.log_file)
    logFile  = open(logFilepath, 'a')

    model.eval()
    losses = AverageMeter()
    num_batch = len(valdataloader)



    i = 0
    for X, Y in valdataloader:
        X = X.to(device)
        
        i += 1 
        # Inference model to generate heatmap
        Y_pred = model(X)

        # Contrastive loss
        loss = loss_func2(S_true=Y,S_pred=Y_pred, gamma=0.1, num_speaker=2)
    

        losses.update(loss.item())

        if DEBUG:
            tensorboardLogger.log(f"val/loss", loss.item(), epoch*num_batch+i)

   
    message = f"Epoch : {epoch}. Loss validation :{losses.avg}"
    logFile.write(message + "\n")

    tensorboardLogger.log(f"val/loss_avg", losses.avg, epoch)


    return losses.avg


## Visualization
def _put_text(img, text, point, color, thickness):
    img = cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, thickness, cv2.LINE_AA)
    return img

def main(args):
    tensorboardLogger = TensorBoardLogger(root="runs", experiment_name=args.snapshot)

    # Init model
    arch_dict = {"AudioMixtureDataset":AudioMixtureDataset}
    assert args.arch in arch_dict.keys(), f'Backbone should be one of {arch_dict.keys()}'
    print(f'Use backbone :{args.arch}')
    model = arch_dict[args.arch]()
    
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['plfd_backbone'])

    model.to(device)
  

    dataset = AudioMixtureDataset(filename=args.data_file,\
                                 database_dir_path=args.data_dir)
  
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        [{
            'params': model.parameters()
        }],
        lr=args.lr,
        weight_decay=1e-6)

    if args.consin_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_num_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size ,gamma=args.gamma)

    
    # # for im, lm in train_dataset:
    # #     print(type(im), lm.shape)

    if args.mode == 'train':
        for epoch in range(args.max_num_epoch):

            # Train    
            train_one_epoch(dataloader, model, optimizer, epoch, args, tensorboardLogger)

            save_checkpoint({
                'epoch': epoch,
                'plfd_backbone': model.state_dict()
            }, filename=f'{args.snapshot}/epoch_{epoch}.pth.tar')
            scheduler.step()
  




def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    parser.add_argument('--snapshot', default='./ckpt_170420221_ghostnet_regression_lmks', type=str, metavar='PATH')
    parser.add_argument('--log_file', default="log.txt", type=str)
    parser.add_argument('--data_file', default="log.txt", type=str)
    parser.add_argument('--data_dir', default="log.txt", type=str)
    parser.add_argument('--train_batchsize', default=16, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--step_size', default=30, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--resume', default="", type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--arch', default="AudioOnlyModel", type=str)
    parser.add_argument('--vis_dir', default="./vis", type=str)
    parser.add_argument('--consin_lr_scheduler', default=0, type=int) # Default is stepLR
    parser.add_argument('--max_num_epoch', default=100, type=int) # Max number of epochs

    











    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

          
