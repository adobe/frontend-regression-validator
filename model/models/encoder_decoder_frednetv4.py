import torch.nn as nn
from torchsummary import summary
from torchvision.models import resnet50
import torch
from models.DoubleUpconv import DoubleUpconv
from models.OutConv import OutConv

global_residuals = [0, 0, 0, 0]
counter = 0


def hook(module, input, output):
    global counter
    counter += 1
    if counter == 3:
        global_residuals[0] = output
    if counter == 41:
        global_residuals[1] = output
    if counter == 85:
        global_residuals[2] = output
    if counter == 149:
        global_residuals[3] = output


class FREDv4Encoder(nn.Module):
    def __init__(self):
        super(FREDv4Encoder, self).__init__()
        self.residuals = []
        self.encoder = nn.Sequential(*list(resnet50(pretrained=True, progress=True).children())[:-2])
        for param in self.encoder.parameters():
            param.requires_grad = True

        for layer_name, layer in self.encoder.named_modules():
            layer.register_forward_hook(hook)

    def forward(self, x, return_residuals=False):
        global global_residuals
        if return_residuals:
            return self.encoder(x), global_residuals
        return self.encoder(x), None


class FREDv4Decoder(nn.Module):
    def __init__(self, out_channels=3, use_residuals=False, out_layer=None):
        super(FREDv4Decoder, self).__init__()
        self.use_residuals = use_residuals
        if self.use_residuals:
            # 16x16
            self.upconv1 = DoubleUpconv(in_channels=2048, out_channels=256)
            # 32x32
            self.upconv2 = DoubleUpconv(in_channels=768, out_channels=256)
            # 64x64
            self.upconv3 = DoubleUpconv(in_channels=512, out_channels=128)
            # 128x128
            self.upconv4 = DoubleUpconv(in_channels=256, out_channels=64)
            # 256x256
            self.upconv5 = DoubleUpconv(in_channels=128, out_channels=out_channels)
            # 512x512
        else:
            # 16x16
            self.upconv1 = DoubleUpconv(in_channels=2048, out_channels=256)
            # 32x32
            self.upconv2 = DoubleUpconv(in_channels=256, out_channels=128)
            # 64x64
            self.upconv3 = DoubleUpconv(in_channels=128, out_channels=64)
            # 128x128
            self.upconv4 = DoubleUpconv(in_channels=64, out_channels=32)
            # 256x256
            self.upconv5 = DoubleUpconv(in_channels=32, out_channels=out_channels)
            # 512x512



    def forward(self, x, residuals):
        if self.use_residuals:
            x1 = residuals[0]
            x2 = residuals[1]
            x3 = residuals[2]
            x4 = residuals[3]

            x = self.upconv1(x)
            x = torch.relu(x)
            x = torch.cat([x, x4], dim=1)

            x = self.upconv2(x)
            x = torch.relu(x)
            x = torch.cat([x, x3], dim=1)

            x = self.upconv3(x)
            x = torch.relu(x)
            x = torch.cat([x, x2], dim=1)

            x = self.upconv4(x)
            x = torch.relu(x)
            x = torch.cat([x, x1], dim=1)
            x = self.upconv5(x)
            x = torch.sigmoid(x)


        else:
            x = self.upconv1(x)
            x = torch.relu(x)

            x = self.upconv2(x)
            x = torch.relu(x)

            x = self.upconv3(x)
            x = torch.relu(x)

            x = self.upconv4(x)
            x = torch.relu(x)

            x = self.upconv5(x)
            x = torch.sigmoid(x)


        global counter
        counter = 0
        return x
