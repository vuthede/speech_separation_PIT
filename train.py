

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
import math
from datasets.VoiceMixtureDataSet import AudioMixtureDataset
from losses.pit_loss import permutation_invariant_training_loss
from datasets.LibriMixDataSet import LibriMix

from logger.TensorboardLogger import TensorBoardLogger
from tqdm import tqdm
from lib import utils
import soundfile as sf
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


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



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f'Save checkpoint to {filename}')


def decode(F_mix, crm):
    F_mix = F_mix.permute(1,2,0)
    F = utils.fast_icRM_torch(F_mix, crm)
    source = utils.fast_istft_torch(F,power=False)
    return source

def train_one_epoch(traindataloader, model, optimizer, epoch, args=None, tensorboardLogger=None):
    model.train()
    losses= AverageMeter()
    num_batch = len(traindataloader)
    i = 0
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")


    with tqdm(traindataloader, unit="batch") as tepoch:
        for (X, Y) in tepoch:
            i += 1
        
            tepoch.set_description(f"Epoch {epoch}")
            
            X = X.to(device)
        
            # Inference model to generate heatmap
            Y_pred = model(X)
            Y_pred = Y_pred 

            #  Permutation Invariant Training Loss
            loss = permutation_invariant_training_loss(S_true=Y.cpu(),S_pred=Y_pred.cpu(), num_speaker=2, only_real=False)

            ######## Pairwise_neg_sisdr #############
            est_source = torch.zeros(X.shape[0], 2, 48000)
            source = torch.zeros(X.shape[0], 2, 48000)
            for b in range(X.shape[0]):
                for j in range(2):
                    est_source[b,j,:] = decode(X[b,:,:,:].cpu(), Y_pred[b,:,:,:,j].cpu())
                    source[b,j,:] =  decode(X[b,:,:,:].cpu(), Y[b,:,:,:,j].cpu())
            est_source = torch.Tensor(est_source)
            source = torch.Tensor(source)
            loss_sir,_ = loss_func(est_source, source, return_est=True)
            ##########################################

            optimizer.zero_grad()
            if args.use_pairwise_neg_sisdr_loss_for_backward:
                loss_sir.backward()
            else:
                loss.backward()
            optimizer.step()
            losses.update(loss.item())

            # if DEBUG:
                # tensorboardLogger.log(f"train/loss", loss.item(), epoch*num_batch+i)

            
            tepoch.set_postfix(loss=loss.item(), loss_sir=loss_sir.item())

        # Log train averge loss in each dataset
        # tensorboardLogger.log(f"train/loss_avg", losses.avg, epoch)
    return losses.avg

def vis_batch(X, Y, Y_pred, batch_number=0, output_viz="./viz"):
    """
    X: Bx2xHxW
    Y: BxHxWx2x2
    Y_pred: BxHxWx2x2
    """

    if not os.path.isdir(output_viz):
        os.makedirs(output_viz)

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y_pred = Y_pred.detach().cpu().numpy()
    
    i = 0
    sample_rate = 8000
    for Xi, Yi, Yi_p in zip(X, Y, Y_pred):
        i += 1
        Xi = np.transpose(Xi, (1,2,0)) # HxWx2

        # Audio mix
        audio_mix = utils.fast_istft(Xi)
        sf.write(f"{output_viz}/batch{batch_number}_sample{i}_mix.wav", audio_mix, sample_rate)

        # Person1 GT
        F1 = utils.fast_icRM(Xi, Yi[:,:,:,0])
        T1 = utils.fast_istft(F1,power=False)
        sf.write(f"{output_viz}/batch{batch_number}_sample{i}_person1_gt.wav", T1, sample_rate)
        
        # Person1 pred
        # Yi_p[:,:,1,0] = Yi[:,:,1,0]
        F1 = utils.fast_icRM(Xi, Yi_p[:,:,:,0])
        T1 = utils.fast_istft(F1,power=False)
        sf.write(f"{output_viz}/batch{batch_number}_sample{i}_person1_pred.wav", T1, sample_rate)

        # Person2 GT
        F2 = utils.fast_icRM(Xi, Yi[:,:,:,1])
        T2 = utils.fast_istft(F2,power=False)
        sf.write(f"{output_viz}/batch{batch_number}_sample{i}_person2_gt.wav", T2, sample_rate)
        
        # Person2 pred
        # Yi_p[:,:,1,1] = Yi[:,:,1,0]
        F2 = utils.fast_icRM(Xi, Yi_p[:,:,:,1])
        T2 = utils.fast_istft(F2,power=False)
        sf.write(f"{output_viz}/batch{batch_number}_sample{i}_person2_pred.wav", T2, sample_rate)


def validate(valdataloader, model, optimizer, epoch, args, tensorboardLogger=None):
    if not os.path.isdir(args.snapshot):
        os.makedirs(args.snapshot)

    logFilepath  = os.path.join(args.snapshot, args.log_file)
    logFile  = open(logFilepath, 'a')

    model.eval()
    losses = AverageMeter()
    losses_sir = AverageMeter()

    num_batch = len(valdataloader)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")



    i = 0
    for X, Y in valdataloader:
        X = X.to(device)
        X_origin = torch.clone(X)
        print(f'Batch {i} in total {len(valdataloader)}')
        i += 1 

        # Inference model to generate heatmap
        Y_pred = model(X)

        #  Permutation Invariant Training Loss
        loss = permutation_invariant_training_loss(S_true=Y.cpu(),S_pred=Y_pred.cpu(), num_speaker=2, only_real=False)

        ######## Pairwise_neg_sisdr #############
        est_source = torch.zeros(X.shape[0], 2, 48000)
        source = torch.zeros(X.shape[0], 2, 48000)
        for b in range(X.shape[0]):
            for j in range(2):
                est_source[b,j,:] = decode(X[b,:,:,:].cpu(), Y_pred[b,:,:,:,j].cpu())
                source[b,j,:] =  decode(X[b,:,:,:].cpu(), Y[b,:,:,:,j].cpu())
        est_source = torch.Tensor(est_source)
        source = torch.Tensor(source)
        loss_sir,_ = loss_func(est_source, source, return_est=True)
        ##########################################

        losses.update(loss.item())
        losses_sir.update(loss_sir.item())

        # if DEBUG:
            # tensorboardLogger.log(f"val/loss", loss.item(), epoch*num_batch+i)

        vis_batch(X_origin, Y, Y_pred, batch_number=i, output_viz=args.output_viz)
   
    message = f"Epoch : {epoch}. Loss validation :{losses.avg}. Loss_sir:{losses_sir.avg}"
    logFile.write(message + "\n")

    # tensorboardLogger.log(f"val/loss_avg", losses.avg, epoch)

    return losses.avg



def main(args):
    # pass
    # tensorboardLogger = TensorBoardLogger(root="runs", experiment_name=args.snapshot)
    tensorboardLogger = None

    # Init model
    arch_dict = {"AudioOnlyModel":AudioOnlyModel}
    assert args.arch in arch_dict.keys(), f'Backbone should be one of {arch_dict.keys()}'
    print(f'Use backbone :{args.arch}')
    model = arch_dict[args.arch]()
    
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['plfd_backbone'])

    model.to(device)
  
    # Dataset and Dataloaders
    train_set = LibriMix(
        csv_dir=f"{args.data_dir}/metadata/",
        task="sep_clean",
        sample_rate=16000,
        n_src=2,
        segment=3,
        return_id=True,
        set_type='train'
    )  

    val_set = LibriMix(
        csv_dir=f"{args.data_dir}/metadata/",
        task="sep_clean",
        sample_rate=16000,
        n_src=2,
        segment=3,
        return_id=True,
        set_type='val'
    )  
  
    dataloader = DataLoader(
        train_set,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=8,
        drop_last=True)

    valdataloader = DataLoader(
        val_set,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=8,
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

    
    # Training and validation
    if args.mode == 'train':
        for epoch in range(args.max_num_epoch):

            # Train    
            train_one_epoch(dataloader, model, optimizer, epoch, args, tensorboardLogger)
            validate(valdataloader, model, optimizer, epoch, args, tensorboardLogger)

            if not os.path.isdir(args.snapshot):
                os.makedirs(args.snapshot)

            save_checkpoint({
                'epoch': epoch,
                'plfd_backbone': model.state_dict()
            }, filename=f'{args.snapshot}/epoch_{epoch}.pth.tar')
            scheduler.step()
    else:
        validate(valdataloader, model, optimizer, 0, args, tensorboardLogger)



def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    parser.add_argument('--snapshot', default='./ckpt_170420221_ghostnet_regression_lmks', type=str, metavar='PATH')
    parser.add_argument('--log_file', default="log.txt", type=str)
    parser.add_argument('--output_viz', default="./viz", type=str)
    parser.add_argument('--data_dir', default="MiniLibriMix", type=str)
    parser.add_argument('--train_batchsize', default=4, type=int)
    parser.add_argument('--val_batchsize', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--step_size', default=30, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--resume', default="", type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--arch', default="AudioOnlyModel", type=str)
    parser.add_argument('--vis_dir', default="./vis", type=str)
    parser.add_argument('--consin_lr_scheduler', default=0, type=int) # Default is stepLR
    parser.add_argument('--max_num_epoch', default=100, type=int) # Max number of epochs
    parser.add_argument('--use_pairwise_neg_sisdr_loss_for_backward', default=0, type=int)


    











    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

          
