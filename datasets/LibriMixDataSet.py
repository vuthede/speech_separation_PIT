from numpy.core.fromnumeric import transpose
import torch
import librosa
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import sys
sys.path.insert(0, "..")
from lib import utils





class LibriMix(Dataset):

    def __init__(
        self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, segment=3, return_id=False, set_type="train"
    ):
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        # Get the csv corresponding to the task
        if task == "enh_single":
            md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "enh_both":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
            md_clean_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))
        elif task == "sep_clean":
            if set_type == "train":
                md_file = [f for f in os.listdir(csv_dir) if "train_mix_clean" in f][0]
            elif set_type == "val":
                md_file = [f for f in os.listdir(csv_dir) if "val_mix_clean" in f][0]
            else:
                raise NotImplementedError

            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "sep_noisy":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        self.n_src = n_src

    def __len__(self):
        return len(self.df)

    def __encode_waves(self, mixture, sources):
        # transfrom data via STFT and several preprocessing function
        F_list = []
        for i in range(len(sources)):
            F = utils.fast_stft(sources[i],power=False)
            F_list.append(F)
        F_mix = utils.fast_stft(mixture,power=False)

        # create cRM for each speaker and fill into y_sample
        cRM_list = []
        for i in range(len(sources)):
            crm = utils.fast_cRM(F_list[i],F_mix)
            crm  = torch.Tensor(crm)
            cRM_list.append(crm)

        cRM_list = torch.stack(cRM_list, dim=-1)


        return F_mix, cRM_list



    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture
        if "enh_both" in self.task:
            mix_clean_path = self.df_clean.iloc[idx]["mixture_path"]
            s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)

        else:
            # Read sources
            for i in range(self.n_src):
                source_path = row[f"source_{i + 1}_path"]
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
                sources_list.append(s)
        # Read the mixture
        # print("Clean path", mixture_path)

        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if not self.return_id:
            return mixture, sources
        # 5400-34479-0005_4973-24515-0007.wav
        id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")

        F_mix, cRM = self.__encode_waves(mixture, sources)
        F_mix = np.transpose(F_mix, (2, 0,1))
        F_mix = torch.Tensor(F_mix) 
        return F_mix, cRM
