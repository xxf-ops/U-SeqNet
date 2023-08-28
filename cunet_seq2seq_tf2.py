import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional

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
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))  # update
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)  # reset之后a
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

# class Decoder(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(Decoder, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_relu = nn.Sequential(
#             nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
#             # nn.ReLU(inplace=True)
#             nn.ReLU()
#         )
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x1 = torch.cat((x1, x2), dim=1)
#         x1 = self.conv_relu(x1)
#         return x1

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up = nn.Sequential(LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))

        # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = Block(dim=out_channels)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.conv1(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = self.conv2(x)
        # print(x.shape)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(

            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(in_channels, num_classes, kernel_size=4, stride=4)
        )


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class Unet_seq2seq(nn.Module):
    def __init__(self, input_channels, out_channels, depths: list = [3, 3, 9, 3], dims: list = [96, 192, 384, 768], drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,):
        super(Unet_seq2seq, self).__init__()
        self.out_channels = out_channels
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        self.stem2 = nn.Sequential(nn.Conv2d(self.out_channels, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            # print(i)
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, 0, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

        self.eclstm1 = ConvLSTMCell(96, 96, 3)
        self.dclstm1 = ConvLSTMCell(96, 96, 3)
        self.eclstm2 = ConvLSTMCell(192, 192, 3)
        self.dclstm2 = ConvLSTMCell(192, 192, 3)
        self.eclstm3 = ConvLSTMCell(384, 384, 3)
        self.dclstm3 = ConvLSTMCell(384, 384, 3)
        self.eclstm4 = ConvLSTMCell(768, 768, 3)
        self.dclstm4 = ConvLSTMCell(768, 768, 3)
        self.eclstm1.init_hidden(1, 96, (64, 64))
        self.dclstm1.init_hidden(1, 96, (64, 64))
        self.eclstm2.init_hidden(1, 192, (32, 32))
        self.dclstm2.init_hidden(1, 192, (32, 32))
        self.eclstm3.init_hidden(1, 384, (16, 16))
        self.dclstm3.init_hidden(1, 384, (16, 16))
        self.eclstm4.init_hidden(1, 768, (8, 8))
        self.dclstm4.init_hidden(1, 768, (8, 8))
        # self.decode3 = Decoder(768, 384 + 384, 384)
        # self.decode2 = Decoder(384, 192 + 192, 192)
        # self.decode1 = Decoder(192, 96 + 96, 96)
        # self.decode0 = nn.Sequential(
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #     nn.Conv2d(96, 48, kernel_size=3, padding=1, bias=False),
        #     nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False)
        # )
        self.decode3 = Up(768, 384)
        self.decode2 = Up(384, 192)
        self.decode1 = Up(192, 96)


        # self.decode0 = OutConv(96, 96)

        # self.conv_last = nn.Conv2d(96, out_channels, 1)
        self.conv_last = OutConv(96, out_channels)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, input, input2, cloud_mask, out_step, device=torch.device("cpu")):  # input batch_size,step,channels,height, width
        output = torch.zeros(input.shape[0], out_step, self.out_channels, input.shape[3], input.shape[4], device=device)
        cloud_mask2 = torch.logical_not(cloud_mask)
        # input22 = input2.clone()
        (eh1, ec1) = self.eclstm1.init_hidden(input.shape[0], 96, (64, 64), device=device)
        self.dclstm1.init_hidden(input.shape[0], 96, (64, 64), device=device)
        (eh2, ec2) = self.eclstm2.init_hidden(input.shape[0], 192, (32, 32), device=device)
        self.dclstm2.init_hidden(input.shape[0], 192, (32, 32), device=device)
        (eh3, ec3) = self.eclstm3.init_hidden(input.shape[0], 384, (16, 16), device=device)
        self.dclstm3.init_hidden(input.shape[0], 384, (16, 16), device=device)
        (eh4, ec4) = self.eclstm4.init_hidden(input.shape[0], 768, (8, 8), device=device)
        self.dclstm4.init_hidden(input.shape[0], 768, (8, 8), device=device)


        for i in range(input.shape[1]):
            # print(f'encoder{i}')
            e1 = self.downsample_layers[0](input[:, i])  # 64,128,128
            e1 = self.stages[0](e1)
            eh1, ec1 = self.eclstm1(e1, eh1, ec1)

            e2 = self.downsample_layers[1](e1)
            e2 = self.stages[1](e2)
            eh2, ec2 = self.eclstm2(e2, eh2, ec2)

            e3 = self.downsample_layers[2](e2)  # 64,64,64
            e3 = self.stages[2](e3)
            eh3, ec3 = self.eclstm3(e3, eh3, ec3)

            e4 = self.downsample_layers[3](e3)
            e4 = self.stages[3](e4)
            eh4, ec4 = self.eclstm4(e4, eh4, ec4)

        dh1, dc1, dh2, dc2, dh3, dc3, dh4, dc4= eh1, ec1, eh2, ec2, eh3, ec3, eh4, ec4


        for j in range(out_step):
            if j == 0:
                dh4, dc4 = self.dclstm4(dh4, dh4, dc4)
                dh3, dc3 = self.dclstm3(dh3, dh3, dc3)
                dh2, dc2 = self.dclstm2(dh2, dh2, dc2)
                dh1, dc1 = self.dclstm1(dh1, dh1, dc1)
            else:
                input2[:, j - 1][cloud_mask2[:, j - 1]] = output[:, j - 1][cloud_mask2[:, j - 1]]
                input22 =input2.clone() #得用复制的，不然会报错inplace，说是不能就地修改（inplace operation），导致版本不匹配（version mismatch），从而无法正确计算梯度。
                l1 = self.stem2(input22[:, j-1])  # 64,128,128
                l1 = self.stages[0](l1)

                l2 = self.downsample_layers[1](l1)
                l2 = self.stages[1](l2)
                l3 = self.downsample_layers[2](l2)  # 64,64,64
                l3 = self.stages[2](l3)
                l4 = self.downsample_layers[3](l3)
                l4 = self.stages[3](l4)

                dh4, dc4 = self.dclstm4(l4, dh4, dc4)
                dh3, dc3 = self.dclstm3(l3, dh3, dc3)
                dh2, dc2 = self.dclstm2(l2, dh2, dc2)
                dh1, dc1 = self.dclstm1(l1, dh1, dc1)

            d3 = self.decode3(dh4, dh3)  # 256,32,32
            d2 = self.decode2(d3, dh2)  # 128,64,64
            d1 = self.decode1(d2, dh1)  # 64,128,128
            # d0 = self.decode0(d1)  # 64,256,256
            output[:, j] = self.conv_last(d1)  # 1,256,256
        return torch.sigmoid(output)


if __name__ == "__main__":
    useqnet = Unet_seq2seq(input_channels=2, out_channels=7)
    input = torch.randn((1, 4, 2, 256, 256))
    input2 = torch.randn((1, 22, 7, 256, 256))
    cloud_mask = torch.randn((1, 22, 7, 256, 256))
    output = useqnet(input=input, input2=input2, cloud_mask=cloud_mask, out_step=input2.shape[1])
    print(output.shape)

