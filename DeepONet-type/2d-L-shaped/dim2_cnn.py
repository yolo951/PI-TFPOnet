import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_Tanh, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.tanh(x)
        return x


class encoder_decoder(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(encoder_decoder, self).__init__()

        # Encoder
        self.convblock1 = ConvBlock(1, dim1, kernel_size=3, stride=2, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=2, padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock3_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock5 = ConvBlock(dim4, dim5, kernel_size=3, stride=1, padding=1)


        # Decoder
        self.deconv1_1 = DeconvBlock(dim5, dim4, kernel_size=2, stride=2, padding=0)
        self.deconv1_2 = ConvBlock(dim4, dim4)
        self.deconv2_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim3, dim3)
        self.deconv3_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim2, dim2)
        self.deconv4_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim1, dim1)
        self.deconv5 = ConvBlock_Tanh(dim1, 4)

    def forward(self, x):

        x = x.unsqueeze(1)

        # encoder
        x = self.convblock1(x)  # (batch, 16, 17, 17)
        x = self.convblock2_1(x)  # (batch, 32, 9, 9)
        x = self.convblock2_2(x)  # (batch, 32, 9, 9)
        x = self.convblock3_1(x)  # (batch, 64, 5, 5)
        x = self.convblock3_2(x)  # (batch, 64, 5, 5)
        x = self.convblock4_1(x)  # (batch, 128, 3, 3)
        x = self.convblock4_2(x)  # (batch, 128, 3, 3)
        x = self.convblock5(x)  # (batch, 256, 2, 2)

        # decoder
        x = self.deconv1_1(x)  # (batch, 128, 3, 3)
        x = self.deconv1_2(x)  # (batch, 128, 3, 3)
        x = self.deconv2_1(x)  # (batch, 64, 6, 6)
        x = self.deconv2_2(x)  # (batch, 64, 6, 6)
        x = self.deconv3_1(x)  # (batch, 32, 12, 12)
        x = self.deconv3_2(x)  # (batch, 32, 12, 12)
        x = self.deconv4_1(x)  # (batch, 16, 24, 24)
        x = self.deconv4_2(x)  # (batch, 16, 24, 24)
        x = self.deconv5(x)  # (batch, 4, 24, 24)


        x = x.permute(0, 2, 3, 1)
        return x





# N = 16
# model = encoder_decoder()

# batch_size = 60
# x = torch.randn(batch_size, N, N)


# with torch.no_grad():
#     test_output = model(x)
#     print(f'Test Output Shape: {test_output.shape}')
