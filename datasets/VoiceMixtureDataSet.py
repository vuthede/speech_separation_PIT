from numpy.core.fromnumeric import transpose
import torch
import librosa
import numpy as np

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

if __name__ == "__main__":
    dataset = AudioMixtureDataset(filename="/home/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt",\
                                 database_dir_path="/home/vuthede/speech_separation/data/audio/audio_database")
    

    print(len(dataset))
    X1, y1 = dataset[0]
    X2, y2 = dataset[0]
    loss = loss_func(y1, y2)
    loss2 = loss_func2(y1.unsqueeze(0), y2.unsqueeze(0))

    print("Loss:", loss, loss2)