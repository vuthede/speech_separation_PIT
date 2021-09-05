"""
\ Mostly ref from here https://github.com/bill9800/speech_separation/blob/master/model/lib/utils.py
  There is some modify like remove tanh_compression in fast_cRM and fast_icRM.
"""

import numpy as np
import librosa
import os 
import itertools
from numpy import inf
import torch

#### Load Data ####
# windowing fft/ifft function
def stft(data, fft_size=512, step_size=160,padding=True):
    # short time fourier transform
    if padding == True:
        # for 16K sample rate data, 48192-192 = 48000
        pad = np.zeros(192,)
        data = np.concatenate((data,pad),axis=0)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
    win_num = (len(data) - fft_size) // step_size
    out = np.ndarray((win_num, fft_size), dtype=data.dtype)
    for i in range(win_num):
        left = int(i * step_size)
        right = int(left + fft_size)
        out[i] = data[left: right] * window
    F = np.fft.rfft(out, axis=1)
    return F

def istft(F, fft_size=512, step_size=160,padding=True):
    # inverse short time fourier transform
    data = np.fft.irfft(F, axis=-1)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
    number_windows = F.shape[0]
    T = np.zeros((number_windows * step_size + fft_size))
    for i in range(number_windows):
        head = int(i * step_size)
        tail = int(head + fft_size)
        T[head:tail] = T[head:tail] + data[i, :] * window
    if padding == True:
        T = T[:48000]
    return T

# combine FFT bins to mel frequency bins
def mel2freq(mel_data,sr,fft_size,n_mel,fmax=8000):
    matrix= librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    return np.dot(mel_data,matrix)

def freq2mel(f_data,sr,fft_size,n_mel,fmax=8000):
    pre_matrix = librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    matrix = pre_matrix.T / np.sum(pre_matrix.T,axis=0)
    return np.dot(f_data,matrix)

# directly time to mel domain transformation
def time_to_mel(data,sr,fft_size,n_mel,step_size,fmax=8000):
    F = stft(data,fft_size,step_size)
    M = freq2mel(F,sr,fft_size,n_mel,fmax=8000)
    return M

def mel_to_time(M,sr,fft_size,n_mel,step_size,fmax=8000):
    F = mel2freq(M,sr,fft_size,n_mel)
    T = istft(F,fft_size,step_size)
    return T

def real_imag_expand(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D

def real_imag_shrink(F,dim='new'):
    # dim = 'new' or 'same'
    # shrink the complex data to combine real and imag number
    F_shrink = np.zeros((F.shape[0], F.shape[1]))
    if dim =='new':
        F_shrink = F[:,:,0] + F[:,:,1]*1j
    if dim =='same':
        F_shrink = F[:,::2] + F[:,1::2]*1j
    return F_shrink

def power_law(data,power=0.6):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data>=0] = 1
    mask[data<0] = -1
    data = np.power(np.abs(data),power)
    data = data*mask
    return data

def fast_stft(data,power=False,**kwargs):
    # directly transform the wav to the input
    # power law = A**0.3 , to prevent loud audio from overwhelming soft audio
    if power:
        data = power_law(data)
    return real_imag_expand(stft(data))

def fast_istft(F,power=False,**kwargs):
    # directly transform the frequency domain data to time domain data
    # apply power law
    T = istft(real_imag_shrink(F))
    if power:
        T = power_law(T,(1.0/0.6))
    return T

def fast_istft_torch(F,power=False,**kwargs):
    # directly transform the frequency domain data to time domain data
    # apply power law
    T = istft_torch(real_imag_shrink_torch(F))
    if power:
        T = power_law_torch(T,(1.0/0.6))
    return T


def real_imag_shrink_torch(F,dim='new'):
    # dim = 'new' or 'same'
    # shrink the complex data to combine real and imag number
    F_shrink = torch.zeros((F.shape[0], F.shape[1]))
    if dim =='new':
        F_shrink = F[:,:,0] + F[:,:,1]*1j
    if dim =='same':
        F_shrink = F[:,::2] + F[:,1::2]*1j
    return F_shrink

def istft_torch(F, fft_size=512, step_size=160,padding=True):
    # inverse short time fourier transform
    data = torch.fft.irfft(F, axis=-1)
    # padding hanning window 512-400 = 112
    window = torch.cat((torch.zeros((56,)),torch.hann_window(fft_size-112, False),torch.zeros((56,))),axis=0)
    number_windows = F.shape[0]
    T = torch.zeros((number_windows * step_size + fft_size))
    for i in range(number_windows):
        head = int(i * step_size)
        tail = int(head + fft_size)
        T[head:tail] = T[head:tail] + data[i, :] * window
    if padding == True:
        T = T[:48000]
    return T


def generate_cRM(Y,S):
    '''

    :param Y: mixed/noisy stft
    :param S: clean stft
    :return: structed cRM
    '''
    M = np.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = np.multiply(Y[:,:,0],S[:,:,0])+np.multiply(Y[:,:,1],S[:,:,1])
    square_real = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_real = np.divide(M_real,square_real+epsilon)
    M[:,:,0] = M_real
    # imaginary part
    M_img = np.multiply(Y[:,:,0],S[:,:,1])-np.multiply(Y[:,:,1],S[:,:,0])
    square_img = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_img = np.divide(M_img,square_img+epsilon)
    M[:,:,1] = M_img
    return M

def cRM_tanh_compress(M,K=10,C=0.1):
    '''
    Recall that the irm takes on vlaues in the range[0,1],compress the cRM with hyperbolic tangent
    :param M: crm (298,257,2)
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''

    numerator = 1-np.exp(-C*M)
    numerator[numerator == inf] = 1
    numerator[numerator == -inf] = -1
    denominator = 1+np.exp(-C*M)
    denominator[denominator == inf] = 1
    denominator[denominator == -inf] = -1
    crm = K * np.divide(numerator,denominator)

    return crm

def cRM_tanh_recover(O,K=10,C=0.1):
    '''

    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    '''

    numerator = K-O
    denominator = K+O
    M = -np.multiply((1.0/C),np.log(np.divide(numerator,denominator)+1.1))

    return M

def cRM_tanh_recover_torch(O,K=10,C=0.1):
    '''

    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    '''

    numerator = K-O
    denominator = K+O
    M = -(1.0/C)*torch.log(torch.divide(numerator,denominator+1e-6)+1.1)

    return M

def fast_cRM(Fclean,Fmix,K=10,C=0.1):
    '''

    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    # crm = cRM_tanh_compress(M,K,C)
    crm = M
    return crm

def fast_icRM(Y,crm,K=10,C=0.1):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    # M = cRM_tanh_recover(crm,K,C)
    M = crm
    S = np.zeros(np.shape(M))
    # print(S.shape, M.shape, Y.shape)
    S[:,:,0] = np.multiply(M[:,:,0],Y[:,:,0])-np.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = np.multiply(M[:,:,0],Y[:,:,1])+np.multiply(M[:,:,1],Y[:,:,0])
    return S

def fast_icRM_torch(Y,crm,K=10,C=0.1):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    # M = cRM_tanh_recover_torch(crm,K,C)
    # print("YOoooooooooooooooooooooooo M sum:", M.sum())
    M = crm
    S = torch.zeros(M.shape)
    # print(S.shape, M.shape, Y.shape)
    S[:,:,0] = torch.multiply(M[:,:,0],Y[:,:,0])-torch.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = torch.multiply(M[:,:,0],Y[:,:,1])+torch.multiply(M[:,:,1],Y[:,:,0])
    return S





























