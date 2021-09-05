from numpy.core.fromnumeric import transpose
import torch
import librosa
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader


class AudioMixtureDataset(torch.utils.data.Dataset):
    def __init__(self, filename, database_dir_path, Xdim=(298, 257, 2), ydim=(298, 257, 2, 2), batch_size=4, shuffle=True):
        'Initialization'
        self.filename = self.__loadcontent(filename)
        self.Xdim = Xdim
        self.ydim = ydim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.database_dir_path = database_dir_path

    def __loadcontent(self, filename):
        with open(filename, 'r') as t:
            trainfile = t.readlines()
            return trainfile

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.filename)

    def __getitem__(self, index):
        'Generate one batch of data'
        filename_temp = self.filename[index]
        X, y = self.__data_generation(filename_temp)
        return X, y

    def __data_generation(self, filename_temp):
        # Initialization
        y = np.empty((*self.ydim,))

        # Generate data
        ID = filename_temp
        info = ID.strip().split(' ')
        X = np.load(self.database_dir_path+'/mix/' + info[0])

        for j in range(2):
            y[:, :, :, j] = np.load(self.database_dir_path+'/crm/' + info[j + 1])

        X = np.transpose(X, (2, 0,1))
        return torch.FloatTensor(X), torch.FloatTensor(y)


