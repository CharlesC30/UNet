import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu, interpolate, max_pool2d
from partialconv2d import PartialConv2d


# Note: original U-Net paper does not use padding on Conv2d, therefore image size changes after each convolution
class PConvUNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        # Encoder
        # Modified from: https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
        # -------
        self.e11 = PartialConv2d(n_channels, 64, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.e12 = PartialConv2d(64, 64, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e21 = PartialConv2d(64, 128, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.e22 = PartialConv2d(128, 128, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e31 = PartialConv2d(128, 256, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.e32 = PartialConv2d(256, 256, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e41 = PartialConv2d(256, 512, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.e42 = PartialConv2d(512, 512, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e51 = nn.Conv2d(512, 1024, kernel_size=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Decoder
        # self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = PartialConv2d(1024, 512, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d12 = PartialConv2d(512, 512, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = PartialConv2d(512, 256, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d22 = PartialConv2d(256, 256, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = PartialConv2d(256, 128, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d32 = PartialConv2d(128, 128, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = PartialConv2d(128, 64, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d42 = PartialConv2d(64, 64, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # Output layer
        self.outconv = nn.Conv2d(64, n_channels, kernel_size=1)

    @staticmethod
    def encode_layer(input_x, input_mask, conv1, conv2):
        out_e1, mask_e1 = conv1(input_x, mask_in=input_mask)
        x_e1 = relu(out_e1)
        out_e2, mask_e2 = conv2(x_e1, mask_in=mask_e1)
        x_e2 = relu(out_e2)
        return max_pool2d(x_e2, kernel_size=2, stride=2), max_pool2d(mask_e2, kernel_size=2, stride=2)

    def forward(self, x, mask):
        # Encoder
        out_e11, mask_e11 = self.e11(x, mask_in=mask)
        xe11 = relu(out_e11)
        out_e12, mask_e12 = self.e12(xe11, mask_in=mask_e11)
        xe12 = relu(out_e12)
        xp1 = self.pool1(xe12)
        mask_p1 = self.pool1(mask_e12)

        out_e21, mask_e21 = self.e21(xp1, mask_p1)
        xe21 = relu(out_e21)
        out_e22, mask_e22 = self.e22(xe21, mask_e21)
        xe22 = relu(out_e22)
        xp2 = self.pool2(xe22)
        mask_p2 = self.pool2(mask_e22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = interpolate(xe52, scale_factor=2, mode="nearest")
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = interpolate(xd12, scale_factor=2, mode="nearest")
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = interpolate(xd22, scale_factor=2, mode="nearest")
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = interpolate(xd32, scale_factor=2, mode="nearest")
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    