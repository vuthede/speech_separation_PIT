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
        # X = np.empty((self.batch_size, *self.Xdim))
        y = np.empty((*self.ydim,))

        # Generate data
        ID = filename_temp
        info = ID.strip().split(' ')
        X = np.load(self.database_dir_path+'/mix/' + info[0])

        for j in range(2):
            y[:, :, :, j] = np.load(self.database_dir_path+'/crm/' + info[j + 1])

        # assert y[:,:,:,0] != y[:,:,:,1]
        X = np.transpose(X, (2, 0,1))
        return torch.FloatTensor(X), torch.FloatTensor(y)

def loss_func(S_true,S_pred,gamma=0.1,num_speaker=2):
    sum = 0
    for i in range(num_speaker):
        sum += torch.sum(torch.flatten((torch.square(S_true[:,:,:,i]-S_pred[:,:,:,i]))))
        for j in range(num_speaker):
            if i != j:
                sum -= gamma*torch.sum(torch.flatten((torch.square(S_true[:,:,:,i]-S_pred[:,:,:,j]))))

    loss = sum / (num_speaker*298*257*2)
    return loss

def loss_func2(S_true,S_pred,gamma=0.1, num_speaker=2):
    sum_mtr = torch.zeros_like(S_true[:,:,:,:,0])
    for i in range(num_speaker):
        sum_mtr += torch.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,i])
        for j in range(num_speaker):
            if i != j:
                sum_mtr -= gamma*(torch.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,j]))

    loss = torch.mean(torch.flatten(sum_mtr))

    return loss

def loss_func3(S_true,S_pred,num_speaker=2, only_real=False):
    """
    Permuatation invariant training
    https://arxiv.org/pdf/1607.00325.pdf
    """
    def loss_pit_sample(T, P, b):
        k = 0
        loss_pairs = torch.zeros_like(torch.Tensor([0,0,0,0]))
        for i in range(num_speaker):
            for j in range(num_speaker):
                if only_real:
                    loss = torch.sum(torch.square(S_true[b,:,:,0,i]-S_pred[b,:,:,0,j]))
                else:
                    loss = torch.sum(torch.square(S_true[b,:,:,:,i]-S_pred[b,:,:,:,j]))
                loss_pairs[k] = loss
                k += 1

        per1 = loss_pairs[0] + loss_pairs[3]
        per2 = loss_pairs[1] + loss_pairs[2]

        if per1<per2:
            min_loss=per1
        else:
            min_loss=per2
        
        return min_loss



    loss_batch = torch.zeros(S_true.shape[0])

    for b in range(S_true.shape[0]):
        loss_batch[b] = loss_pit_sample(S_true, S_pred, b)
    
    loss = torch.mean(loss_batch) / (num_speaker*298*257*2)
    return loss

if __name__ == "__main__":
    dataset = AudioMixtureDataset(filename="/home/ubuntu/vuthede/audio_database/dataset_train.txt",\
                                 database_dir_path="/home/ubuntu/vuthede/audio_database")
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        drop_last=True)


    print(len(dataloader))
    X1, y1 = next(iter(dataloader))
    X2, y2 = next(iter(dataloader))
    # loss = loss_func(y1, y2)
    # loss2 = loss_func2(y1.unsqueeze(0), y2.unsqueeze(0))
    
    # y1 =  y1.unsqueeze(0)
    # y2 =  y2.unsqueeze(0)
    # y1 = torch.Tensor([y1, y1])
    # y2 = torch.Tensor([y2, y2])


    loss3 = loss_func3(y1, y2)
    print("Y1 ahpe-----------------------:",y1.shape, y2.shape)
    print("Lossssss:",loss3)

    # loss3.backward()
    # print("Loss:", loss, loss2)

    ##### PIT correct ########
    # from itertools import permutations
    # from asteroid.losses import PITLossWrapper
    # from asteroid.losses import pairwise_mse, singlesrc_mse, multisrc_mse
    # n_sources =2
    # perms = list(permutations(range(n_sources)))
    # print(perms)

    # all_losses = torch.stack([multisrc_mse(y1, y2[:, p]) for p in perms])
    # print(all_losses)

    # best_loss_idx = torch.argmin(all_losses)