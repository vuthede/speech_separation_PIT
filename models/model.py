
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def conv_bn(inp, oup, kernel, stride, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class AudioOnlyModel(nn.Module):
    def __init__(self):
        super(AudioOnlyModel, self).__init__()
        self.conv1 = conv_bn(2, 96, kernel=(1, 7), stride=1, padding=(0, 3), dilation=(1,1))
        self.conv2 = conv_bn(96, 96, kernel=(7, 1), stride=1, padding=(3, 0), dilation=(1,1))
        self.conv3 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2, 2), dilation=(1,1))
        self.conv4 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+1*2, 2), dilation=(2,1))
        self.conv5 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+3*2, 2), dilation=(4,1))
        self.conv6 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+7*2, 2), dilation=(8,1))
        self.conv7 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+15*2, 2), dilation=(16,1))
        self.conv8 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+31*2, 2), dilation=(32,1))

        self.conv9 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2, 2), dilation=(1,1))
        self.conv10 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+1*2, 2+1*2), dilation=(2,2))
        self.conv11 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+3*2, 2+3*2), dilation=(4,4))
        self.conv12 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+7*2, 2+7*2), dilation=(8,8))
        self.conv13 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+15*2, 2+15*2), dilation=(16,16))
        self.conv14 = conv_bn(96, 96, kernel=(5, 5), stride=1, padding=(2+31*2, 2+31*2), dilation=(32,32))
        self.conv15 = conv_bn(96, 8, kernel=(1, 1), stride=1, padding=(0, 0), dilation=(1,1))


        self.bilstm = torch.nn.LSTM(2056, hidden_size=200, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(400, 600)
        self.fc2 = torch.nn.Linear(600, 600)
        self.fc3 = torch.nn.Linear(600, 600)
        self.fc_mask = torch.nn.Linear(600, 257*4) # 2 complex mask --> 4 mask




     
        


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        # Convert to Batch x Sequence x Feature
        x = x.permute(0,2,3,1).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        # print(f'conv15: ',x.shape)

        # BiLstm and fc --> mask
        x, _ = self.bilstm(x)
        # print(f'after bilstm: ',x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_mask(x)
        x = x.view(x.shape[0], x.shape[1], 257, 2, 2)

        # print(f'fc_mask: ',x.shape)






        return x

if __name__ == "__main__":

    model = AudioOnlyModel()
    x = torch.rand(2, 2, 298, 257)
    y = model(x)

    print("Y shape: ", y.shape)