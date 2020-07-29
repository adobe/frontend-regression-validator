import torch
import torch.nn as nn
from torchvision.models import resnet50

from models.DoubleUpconvUNet import DoubleUpconvUNet
from models.OutConv import OutConv
from models.residuals import Residuals


class UNetv2SameEncoder(nn.Module):
    def __init__(self):
        super(UNetv2SameEncoder, self).__init__()
        self.encoder = nn.Sequential(*list(resnet50(pretrained=True, progress=True).children())[:-2])
        self.residuals = [Residuals(self.encoder[i]) for i in [1, 3, 5, 6]]

    def forward(self, x, return_residuals=False):
        x = self.encoder(x)
        if return_residuals:
            return x, self.residuals
        return x, None


class UNetv2SameDecoder(nn.Module):
    def __init__(self, out_channels=3, use_residuals=True, out_layer=None):
        super(UNetv2SameDecoder, self).__init__()
        self.use_residuals = use_residuals
        if self.use_residuals:

            self.decoder1 = DoubleUpconvUNet(in_channels=2048, out_channels=1024)

            self.decoder2 = DoubleUpconvUNet(in_channels=2048, out_channels=512)

            self.decoder3 = DoubleUpconvUNet(in_channels=1024, out_channels=256)

            self.decoder4 = DoubleUpconvUNet(in_channels=320, out_channels=64)

            self.decoder5 = DoubleUpconvUNet(in_channels=128, out_channels=out_channels)
        else:

            self.decoder1 = DoubleUpconvUNet(in_channels=2048, out_channels=1024)

            self.decoder2 = DoubleUpconvUNet(in_channels=1024, out_channels=512)

            self.decoder3 = DoubleUpconvUNet(in_channels=512, out_channels=256)

            self.decoder4 = DoubleUpconvUNet(in_channels=256, out_channels=64)

            self.decoder5 = DoubleUpconvUNet(in_channels=128, out_channels=out_channels)

    def forward(self, x, residuals):
        if self.use_residuals:
            x4 = residuals[3]
            x3 = residuals[2]
            x2 = residuals[1]
            x1 = residuals[0]

            y1 = self.decoder1(x)
            y1 = torch.cat((x4.features, y1), dim=1)
            y1 = torch.sigmoid(y1)

            y2 = self.decoder2(y1)
            y2 = torch.cat((x3.features, y2), dim=1)
            y2 = torch.sigmoid(y2)

            y3 = self.decoder3(y2)
            y3 = torch.cat((x2.features, y3), dim=1)
            y3 = torch.sigmoid(y3)

            y4 = self.decoder4(y3)
            y4 = torch.cat((x1.features, y4), dim=1)
            y4 = torch.sigmoid(y4)

            out = self.decoder5(y4)

        else:
            y = self.decoder1(x)
            y = self.decoder2(y)
            y = self.decoder3(y)
            y = self.decoder4(y)
            out = self.decoder5(y)

        out = torch.sigmoid(out)
        return out
