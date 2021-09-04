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
from datasets.VoiceMixtureDataSet import AudioMixtureDataset, loss_func2, loss_func3
from datasets.LibriMixDataSet import LibriMix
from tqdm import tqdm
from lib import utils
import soundfile as sf
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import from_torch_complex, to_torch_complex
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


def decode(F_mix, crm):
    F_mix = F_mix.permute(1,2,0)
    # print("YOOOOOOOOOOOOOOO F_mix sum:", F_mix.sum())
    F = utils.fast_icRM_torch(F_mix, crm)
    # print("YOOOOOOOOOOOOOOO F sum:", F.sum())
    source = utils.fast_istft_torch(F,power=False)
    # print("YOOOOOOOOOOOOOOO source:", source)

    return source

if __name__ == "__main__":
    test_set = LibriMix(
    csv_dir="./MiniLibriMix/metadata/",
    task="sep_clean",
    sample_rate=16000,
    n_src=2,
    segment=3,
    return_id=True,
    set_type='val'
)  # Uses all segment length

    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # mix, sources, ids = test_set[0]
    Xi, Yi = test_set[0]
    
    src1 = decode(Xi, Yi[:,:,:,0])
    src2 = decode(Xi, Yi[:,:,:,1])
    mix = utils.fast_istft_torch(Xi.permute(1,2,0),power=False)
    sf.write("mix.wav", mix,8000)
    sf.write("src1.wav", src1, 8000)
    sf.write("src2.wav", src2,8000)


    # while True:
    #     i = np.random.randint(0, len(test_set)-1)
    #     j = np.random.randint(0, len(test_set)-1)
    #     i = 43
    #     j = 55
        
    #     Xi, Yi = test_set[i]
    #     Xj, Yj = test_set[j]

    #     src1 = decode(Xi, Yi[:,:,:,0])
    #     src2 = decode(Xi, Yi[:,:,:,1])
    #     est = torch.stack([src1, src2], 0).unsqueeze(0)

    #     src3 = decode(Xj, Yj[:,:,:,0])
    #     src4 = decode(Xj, Yj[:,:,:,1])
    #     source = torch.stack([src3, src4], 0).unsqueeze(0)

    #     l = loss_func(est, source)  
    #     print("i:",i, ". j:", j, ".loss:",l)