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

#%%

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

#%%

class Unet_seq2seq(nn.Module):
    def __init__(self, input_channels, out_channels, resnet18_use=False):
        super(Unet_seq2seq, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if resnet18_use else None)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.eclstm1 = ConvLSTMCell(64, 64, 3)
        self.dclstm1 = ConvLSTMCell(64, 64, 3)
        self.eclstm2 = ConvLSTMCell(64, 64, 3)
        self.dclstm2 = ConvLSTMCell(64, 64, 3)
        self.eclstm3 = ConvLSTMCell(128, 128, 3)
        self.dclstm3 = ConvLSTMCell(128, 128, 3)
        self.eclstm4 = ConvLSTMCell(256, 256, 3)
        self.dclstm4 = ConvLSTMCell(256, 256, 3)
        self.eclstm5 = ConvLSTMCell(512, 512, 3)
        self.dclstm5 = ConvLSTMCell(512, 512, 3)
        self.eclstm1.init_hidden(1, 64, (128, 128))
        self.dclstm1.init_hidden(1, 64, (128, 128))
        self.eclstm2.init_hidden(1, 64, (64, 64))
        self.dclstm2.init_hidden(1, 64, (64, 64))
        self.eclstm3.init_hidden(1, 128, (32, 32))
        self.dclstm3.init_hidden(1, 128, (32, 32))
        self.eclstm4.init_hidden(1, 256, (16, 16))
        self.dclstm4.init_hidden(1, 256, (16, 16))
        self.eclstm5.init_hidden(1, 512, (8, 8))
        self.dclstm5.init_hidden(1, 512, (8, 8))
        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 256 + 128, 256)
        self.decode2 = Decoder(256, 128 + 64, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, input, out_step, device=torch.device("cpu")):  # input batch_size,step,channels,height, width
        output = torch.zeros(input.shape[0], out_step, self.out_channels, input.shape[3], input.shape[4], device=device)
        (eh1, ec1) = self.eclstm1.init_hidden(input.shape[0], 64, (128, 128), device=device)
        (eh2, ec2) = self.eclstm2.init_hidden(input.shape[0], 64, (64, 64), device=device)
        (eh3, ec3) = self.eclstm3.init_hidden(input.shape[0], 128, (32, 32), device=device)
        (eh4, ec4) = self.eclstm4.init_hidden(input.shape[0], 256, (16, 16), device=device)
        (eh5, ec5) = self.eclstm5.init_hidden(input.shape[0], 512, (8, 8), device=device)
        for i in range(input.shape[1]):
            print(f'encoder{i}')
            e1 = self.layer1(input[:, i])  # 64,128,128
            print(f'e1mean:{torch.mean(e1).item()},e1var:{torch.var(e1).item()}')
            eh1, ec1 = self.eclstm1(e1, eh1, ec1)
            e2 = self.layer2(e1)  # 64,64,64
            print(f'e2mean:{torch.mean(e2).item()},e2var:{torch.var(e2).item()}')
            eh2, ec2 = self.eclstm2(e2, eh2, ec2)
            e3 = self.layer3(e2)  # 128,32,32
            print(f'e3mean:{torch.mean(e3).item()},e3var:{torch.var(e3).item()}')
            eh3, ec3 = self.eclstm3(e3, eh3, ec3)
            e4 = self.layer4(e3)  # 256,16,16
            print(f'e4mean:{torch.mean(e4).item()},e4var:{torch.var(e4).item()}')
            eh4, ec4 = self.eclstm4(e4, eh4, ec4)
            f = self.layer5(e4)  # 512,8,8
            print(f'e5mean:{torch.mean(f).item()},e5var:{torch.var(f).item()}')
            eh5, ec5 = self.eclstm5(f, eh5, ec5)
            print(f'eh5mean:{torch.mean(eh5).item()},eh5var:{torch.var(eh5).item()}')
            print(f'eh4mean:{torch.mean(eh4).item()},eh4var:{torch.var(eh4).item()}')
            print(f'eh3mean:{torch.mean(eh3).item()},eh3var:{torch.var(eh3).item()}')
            print(f'eh2mean:{torch.mean(eh2).item()},eh2var:{torch.var(eh2).item()}')
            print(f'eh1mean:{torch.mean(eh1).item()},eh1var:{torch.var(ec1).item()}')
            print(f'ec5mean:{torch.mean(ec5).item()},ec5var:{torch.var(ec5).item()}')
            print(f'ec4mean:{torch.mean(ec4).item()},ec4var:{torch.var(ec4).item()}')
            print(f'ec3mean:{torch.mean(ec3).item()},ec3var:{torch.var(ec3).item()}')
            print(f'ec2mean:{torch.mean(ec2).item()},ec2var:{torch.var(ec2).item()}')
            print(f'ec1mean:{torch.mean(ec1).item()},ec1var:{torch.var(ec1).item()}')
        dh1, dc1, dh2, dc2, dh3, dc3, dh4, dc4, dh5, dc5 = eh1, ec1, eh2, ec2, eh3, ec3, eh4, ec4, eh5, ec5
        print('con')
        print(f'dh5mean:{torch.mean(dh5).item()},dh5var:{torch.var(dh5).item()}')
        print(f'dh4mean:{torch.mean(dh4).item()},dh4var:{torch.var(dh4).item()}')
        print(f'dh3mean:{torch.mean(dh3).item()},dh3var:{torch.var(dh3).item()}')
        print(f'dh2mean:{torch.mean(dh2).item()},dh2var:{torch.var(dh2).item()}')
        print(f'dh1mean:{torch.mean(dh1).item()},dh1var:{torch.var(dh1).item()}')
        for j in range(out_step):
            print(f'decoder{j}')
            dh5, dc5 = self.dclstm5(dh5, dh5, dc5)
            print(f'dh5mean:{torch.mean(dh5).item()},dh5var:{torch.var(dh5).item()}')
            dh4, dc4 = self.dclstm4(dh4, dh4, dc4)
            print(f'dh4mean:{torch.mean(dh4).item()},dh4var:{torch.var(dh4).item()}')
            dh3, dc3 = self.dclstm3(dh3, dh3, dc3)
            print(f'dh3mean:{torch.mean(dh3).item()},dh3var:{torch.var(dh3).item()}')
            dh2, dc2 = self.dclstm2(dh2, dh2, dc2)
            print(f'dh2mean:{torch.mean(dh2).item()},dh2var:{torch.var(dh2).item()}')
            dh1, dc1 = self.dclstm1(dh1, dh1, dc1)
            print(f'dh1mean:{torch.mean(dh1).item()},dh1var:{torch.var(dh1).item()}')
            d4 = self.decode4(dh5, dh4)  # 256,16,16
            print(f'd4mean:{torch.mean(d4).item()},d4var:{torch.var(d4).item()}')
            d3 = self.decode3(d4, dh3)  # 256,32,32
            print(f'd3mean:{torch.mean(d3).item()},d3var:{torch.var(d3).item()}')
            d2 = self.decode2(d3, dh2)  # 128,64,64
            print(f'd2mean:{torch.mean(d2).item()},d2var:{torch.var(d2).item()}')
            d1 = self.decode1(d2, dh1)  # 64,128,128
            print(f'd1mean:{torch.mean(d1).item()},d1var:{torch.var(d1).item()}')
            d0 = self.decode0(d1)  # 64,256,256
            print(f'd0mean:{torch.mean(d0).item()},d0var:{torch.var(d0).item()}')
            output[:, j] = self.conv_last(d0)  # 1,256,256
            print(f'outputmean:{torch.mean(output[:, j]).item()},outputvar:{torch.var(output[:, j]).item()}')
        return output

#%%

if __name__ == "__main__":
    # from google.colab import drive
    # drive.mount('/content/drive')
    useqnet = Unet_seq2seq(input_channels=2, out_channels=7, resnet18_use=False)
    # useqnet.load_state_dict(torch.load('/content/drive/MyDrive/model_state_dict/1011_use_nom_useq_01.npy.pkl', map_location=torch.device('cpu')))
    useqnet.eval()
    mean=0.002301445696502924
    std=0.001604937045031913
    area_id = 54
    # input = (torch.load(f'/content/drive/MyDrive/train_data/{area_id}_input.pth')-mean)/std
    input = torch.randn((31,2,256,256))
    with torch.no_grad():
        output = useqnet(input=input[None], out_step=22)
