import torch.nn as nn
from torchsummary import summary
from torchvision.models import resnet50
import torch

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


class FREDv3Encoder(nn.Module):
    def __init__(self):
        super(FREDv3Encoder, self).__init__()
        self.residuals = []
        self.encoder = nn.Sequential(*list(resnet50(pretrained=True, progress=True).children())[:-2])
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        for layer_name, layer in self.encoder.named_modules():
            layer.register_forward_hook(hook)

    def forward(self, x, return_residuals=False):
        global global_residuals
        if return_residuals:
            return self.encoder(x), global_residuals
        return self.encoder(x), None


class FREDv3Decoder(nn.Module):
    def __init__(self, out_channels=3, use_residuals=False, out_layer=None):
        super(FREDv3Decoder, self).__init__()
        self.use_residuals = use_residuals
        if self.use_residuals:
            self.upconv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
            )
            self.upconv2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
            )
            self.upconv3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0)
            )
            self.upconv4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0)
            )
            self.upconv5 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=192, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.upconv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
            )
            self.upconv2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
            )
            self.upconv3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
            )
            self.upconv4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
            )
            self.upconv5 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x, residuals):
        if self.use_residuals:
            x1 = residuals[0]
            x2 = residuals[1]
            x3 = residuals[2]
            x4 = residuals[3]

            x = self.upconv1(x)
            x = torch.tanh(x)
            x = torch.cat([x, x4], dim=1)

            x = self.upconv2(x)
            x = torch.tanh(x)
            x = torch.cat([x, x3], dim=1)

            x = self.upconv3(x)
            x = torch.tanh(x)
            x = torch.cat([x, x2], dim=1)

            x = self.upconv4(x)
            x = torch.tanh(x)
            x = torch.cat([x, x1], dim=1)

            x = self.upconv5(x)
            x = torch.sigmoid(x)

        else:
            x = self.upconv1(x)
            x = torch.tanh(x)

            x = self.upconv2(x)
            x = torch.tanh(x)

            x = self.upconv3(x)
            x = torch.tanh(x)

            x = self.upconv4(x)
            x = torch.tanh(x)

            x = self.upconv5(x)
            x = torch.sigmoid(x)
        global counter
        counter = 0
        return x
