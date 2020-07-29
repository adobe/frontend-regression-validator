import torch.nn as nn
import torch

from models.OutConv import OutConv


class FREDEncoder(nn.Module):
    def __init__(self):
        super(FREDEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        self.mp1 = nn.MaxPool2d(kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.mp2 = nn.MaxPool2d(kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True)
        self.mp3 = nn.MaxPool2d(kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True)
        self.mp4 = nn.MaxPool2d(kernel_size=4)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=2, padding=0, stride=1, bias=True)

    def forward(self, x, return_residuals=False):
        x = self.conv1(x)
        x1 = torch.relu(x)
        x = self.mp1(x1)

        x = self.conv2(x)
        x2 = torch.relu(x)
        x = self.mp2(x2)

        x = self.conv3(x)
        x3 = torch.relu(x)
        x = self.mp3(x3)

        x = self.conv4(x)
        x4 = torch.relu(x)
        x = self.mp4(x4)

        x = self.conv5(x)
        if return_residuals:
            residuals = [x1, x2, x3, x4]
            return x, residuals
        return x, None


class FREDDecoder(nn.Module):
    def __init__(self, out_channels=3, use_residuals=None, out_layer=None):
        super(FREDDecoder, self).__init__()
        self.use_residuals = use_residuals

        if out_layer:
            self.out_layer = OutConv(in_channels=8, out_channels=out_channels)

        if self.use_residuals:
            self.upconv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=2048, kernel_size=1, padding=0, stride=1,
                                              bias=True)
            self.upconv2 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=8, padding=0, stride=1,
                                              bias=True)
            self.upconv3 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, padding=0, stride=4,
                                              bias=True)
            self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, padding=0, stride=4,
                                              bias=True)
            self.upconv5 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=2, padding=0, stride=2,
                                              bias=True)
            self.upconv6 = nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=2, padding=0,
                                              stride=2,
                                              bias=True)
        else:
            self.upconv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=2048, kernel_size=1, padding=0, stride=1,
                                              bias=True)
            self.upconv2 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=8, padding=0, stride=1,
                                              bias=True)
            self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, padding=0, stride=4,
                                              bias=True)
            self.upconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=0, stride=4,
                                              bias=True)
            self.upconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, padding=0, stride=2,
                                              bias=True)
            self.upconv6 = nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=2, padding=0,
                                              stride=2,
                                              bias=True)

    def forward(self, x, residuals):
        if self.use_residuals:
            x1 = residuals[0]
            x2 = residuals[1]
            x3 = residuals[2]
            x4 = residuals[3]

            x = self.upconv1(x)

            x = self.upconv2(x)
            x = torch.tanh(x)
            x = torch.cat([x, x4], dim=1)

            x = self.upconv3(x)
            x = torch.tanh(x)
            x = torch.cat([x, x3], dim=1)

            x = self.upconv4(x)
            x = torch.tanh(x)
            x = torch.cat([x, x2], dim=1)

            x = self.upconv5(x)
            x = torch.tanh(x)

            x = self.upconv6(x)
            x = torch.sigmoid(x)
        else:
            x = self.upconv1(x)

            x = self.upconv2(x)
            x = torch.tanh(x)

            x = self.upconv3(x)
            x = torch.tanh(x)

            x = self.upconv4(x)
            x = torch.tanh(x)

            x = self.upconv5(x)
            x = torch.tanh(x)

            x = self.upconv6(x)
            x = torch.sigmoid(x)
        return x
