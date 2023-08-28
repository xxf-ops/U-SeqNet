#-*- codeing = utf-8 -*-
#@Time : 2023/1/6 22:36
#@Author : 杨箫
#@File : clstm_seq2seq.py
#@Software : PyCharm

#%%

import torch
import torch.nn as nn
import torchvision


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape, device=torch.device("cpu")):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], device=device))
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], device=device))
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], device=device))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.zeros(batch_size, hidden, shape[0], shape[1], device=device),
                torch.zeros(batch_size, hidden, shape[0], shape[1], device=device))


class CLSTM_seq2seq(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CLSTM_seq2seq, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        #     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True))
        self.eclstm = ConvLSTMCell(input_channels, 64, 3)
        self.dclstm = ConvLSTMCell(64, 64, 3)
        self.eclstm.init_hidden(1, 64, (256, 256))
        self.dclstm.init_hidden(1, 64, (256, 256))
        # self.decode = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        # )
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, input, out_step, device=torch.device("cpu")):  # input batch_size,step,channels,height, width
        output = torch.zeros(input.shape[0], out_step, self.out_channels, input.shape[3], input.shape[4], device=device)
        (eh, ec) = self.eclstm.init_hidden(input.shape[0], 64, (256, 256), device=device)
        for i in range(input.shape[1]):
            # e = self.layer1(input[:,i])  # 64,128,128
            e = input[:,i]
            eh, ec = self.eclstm(e, eh, ec)
        dh, dc = eh, ec
        for j in range(out_step):
            dh, dc = self.dclstm(dh, dh, dc)
            # d = self.decode(dh)
            d = dh
            output[:, j] = self.conv_last(d)  # out_channels,256,256
        return output

#%%

if __name__ == "__main__":
    useqnet = CLSTM_seq2seq(input_channels=2, out_channels=3)
    input = torch.randn(1, 15, 2, 256, 256)
    print(input.dtype)
    with torch.no_grad():
        output = useqnet(input=input, out_step=10)
    print(output.shape)  # (1, 10, 3, 256, 256)
