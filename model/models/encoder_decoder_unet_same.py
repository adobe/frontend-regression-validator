import torch
import torch.nn as nn

from models.OutConv import OutConv


class UNetSameEncoder(nn.Module):
    def __init__(self):
        super(UNetSameEncoder, self).__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                      nn.ReLU())

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU())

        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU())

        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU())

        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1),
            nn.ReLU())

    def forward(self, x, return_residuals=False):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        if return_residuals:
            residuals = [x1, x2, x3, x4]
            return x5, residuals
        return x5, None


class UNetSameDecoder(nn.Module):
    def __init__(self, out_channels=3, use_residuals=True, out_layer=None):
        super(UNetSameDecoder, self).__init__()
        self.use_residuals = use_residuals
        if self.use_residuals:

            self.decoder1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder1_conv = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.decoder2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder2_conv = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.decoder3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder3_conv = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.decoder4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder4_conv = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.out = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1),
                                     nn.Sigmoid())





        else:

            self.decoder1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder1_conv = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.decoder2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder2_conv = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.decoder3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder3_conv = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.decoder4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

            self.decoder4_conv = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                               nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                               nn.Sigmoid())

            self.out = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1),
                                     nn.Sigmoid())

    def forward(self, x, residuals):
        if self.use_residuals:
            x4 = residuals[3]
            x3 = residuals[2]
            x2 = residuals[1]
            x1 = residuals[0]

            y1 = self.decoder1(x)
            y1 = torch.cat((x4, y1), dim=1)
            y1 = self.decoder1_conv(y1)

            y2 = self.decoder2(y1)
            y2 = torch.cat((x3, y2), dim=1)
            y2 = self.decoder2_conv(y2)

            y3 = self.decoder3(y2)
            y3 = torch.cat((x2, y3), dim=1)
            y3 = self.decoder3_conv(y3)

            y4 = self.decoder4(y3)
            y4 = torch.cat((x1, y4), dim=1)
            y4 = self.decoder4_conv(y4)

            out = self.out(y4)

        else:
            y = self.decoder1(x)
            y = self.decoder1_conv(y)
            y = self.decoder2(y)
            y = self.decoder2_conv(y)
            y = self.decoder3(y)
            y = self.decoder3_conv(y)
            y = self.decoder4(y)
            y = self.decoder4_conv(y)
            y = self.out(y)

            return y

        return out
