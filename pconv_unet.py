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

        self.e51 = PartialConv2d(512, 1024, kernel_size=1, bias=False, multi_channel=True, return_mask=True)
        self.e52 = PartialConv2d(1024, 1024, kernel_size=1, bias=False, multi_channel=True, return_mask=True)

        # Decoder
        # self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = PartialConv2d(1024 + 512, 512, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d12 = PartialConv2d(512, 512, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = PartialConv2d(512 + 256, 256, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d22 = PartialConv2d(256, 256, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = PartialConv2d(256 + 128, 128, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d32 = PartialConv2d(128, 128, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = PartialConv2d(128 + 64, 64, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)
        self.d42 = PartialConv2d(64, 64, kernel_size=3, padding=1, bias=False, multi_channel=True, return_mask=True)

        # Output layer
        self.outconv = nn.Conv2d(64 + n_channels, n_channels, kernel_size=1)

    @staticmethod
    def pconv_layer(input_x, input_mask, conv1, conv2):
        out_1, mask_1 = conv1(input_x, mask_in=input_mask)
        x_1 = relu(out_1)
        out_2, mask_2 = conv2(x_1, mask_in=mask_1)
        x_2 = relu(out_2)
        return x_2, mask_2

    @staticmethod
    def decode_layer(input_x, input_mask, encoded_x, encoded_mask, conv1, conv2):
        x_upscale = interpolate(input_x, scale_factor=2, mode="nearest")
        mask_upscale = interpolate(input_mask, scale_factor=2, mode="nearest")
        x_skip = torch.cat([x_upscale, encoded_x], dim=1)
        mask_skip = torch.cat([mask_upscale, encoded_mask], dim=1)
        return PConvUNet.pconv_layer(x_skip, mask_skip, conv1, conv2)


    def forward(self, x, mask):
        # Encoder
        x_e1, mask_e1 = self.pconv_layer(x, mask, self.e11, self.e12)
        x_p1 = max_pool2d(x_e1, kernel_size=2, stride=2)
        mask_p1 = max_pool2d(mask_e1, kernel_size=2, stride=2)

        x_e2, mask_e2 = self.pconv_layer(x_p1, mask_p1, self.e21, self.e22)
        x_p2 = max_pool2d(x_e2, kernel_size=2, stride=2)
        mask_p2 = max_pool2d(mask_e2, kernel_size=2, stride=2)

        x_e3, mask_e3 = self.pconv_layer(x_p2, mask_p2, self.e31, self.e32)
        x_p3 = max_pool2d(x_e3, kernel_size=2, stride=2)
        mask_p3 = max_pool2d(mask_e3, kernel_size=2, stride=2)

        x_e4, mask_e4 = self.pconv_layer(x_p3, mask_p3, self.e41, self.e42)
        x_p4 = max_pool2d(x_e4, kernel_size=2, stride=2)
        mask_p4 = max_pool2d(mask_e4, kernel_size=2, stride=2)

        x_e51, mask_e51 = self.e51(x_p4, mask_in=mask_p4)
        x_e52, mask_e52 = self.e52(x_e51, mask_in=mask_e51)

        
        # Decoder
        x_d1, mask_d1 = self.decode_layer(x_e52, mask_e52, x_e4, mask_e4, self.d11, self.d12)

        x_d2, mask_d2 = self.decode_layer(x_d1, mask_d1, x_e3, mask_e3, self.d21, self.d22)

        x_d3, mask_d3 = self.decode_layer(x_d2, mask_d2, x_e2, mask_e2, self.d31, self.d32)

        x_d4, mask_d4 = self.decode_layer(x_d3, mask_d3, x_e1, mask_e1, self.d41, self.d42)

        # Output layer
        out = self.outconv(torch.cat([x_d4, x], dim=1))

        return out
    

        # out_e21, mask_e21 = self.e21(x_p1, mask_p1)
        # xe21 = relu(out_e21)
        # out_e22, mask_e22 = self.e22(xe21, mask_e21)
        # xe22 = relu(out_e22)
        # xp2 = self.pool2(xe22)
        # mask_p2 = self.pool2(mask_e22)

        # xe31 = relu(self.e31(xp2))
        # xe32 = relu(self.e32(xe31))
        # xp3 = self.pool3(xe32)

        # xe41 = relu(self.e41(xp3))
        # xe42 = relu(self.e42(xe41))
        # xp4 = self.pool4(xe42)

        # xe51 = relu(self.e51(xp4))
        # xe52 = relu(self.e52(xe51))