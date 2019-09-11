import torch.nn as nn
from torchvision.models import resnet50
import torch

from models.OutConv import OutConv

global_residuals = [0, 0, 0, 0]
counter = 0


def hook(module, input, output):
    global counter
    counter += 1
    if counter == 3:
        # 64x256x256
        global_residuals[0] = output
    elif counter == 41:
        # 128x128x128
        global_residuals[1] = output
    elif counter == 85:
        # 256x64x64
        global_residuals[2] = output
    elif counter == 149:
        # 512x32x32
        global_residuals[3] = output


class PSPNetEncoder(nn.Module):
    def __init__(self):
        global counter
        counter = 0
        super(PSPNetEncoder, self).__init__()
        self.encoder = nn.Sequential(*list(resnet50(pretrained=True, progress=True).children())[:-2])
        for param in self.encoder.parameters():
            param.requires_grad = False
        for name, module in self.encoder.named_modules():
            counter += 1
            module.register_forward_hook(hook)
        counter = 0

    def forward(self, x, return_residuals=False):
        global global_residuals
        if return_residuals:
            return self.encoder(x), global_residuals
        return self.encoder(x), None


class PSPNetDecoder(nn.Module):
    def __init__(self, out_channels=3, use_residuals=False, out_layer=None):
        super(PSPNetDecoder, self).__init__()
        self.use_residuals = use_residuals
        self.pool1x1 = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)
        self.pool2x2 = nn.AvgPool2d(kernel_size=15, stride=1, padding=0)
        self.pool3x3 = nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        self.pool6x6 = nn.AvgPool2d(kernel_size=11, stride=1, padding=0)
        self.pool9x9 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

        self.conv1x1_1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv1x1_6 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv1x1_9 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.upsample_1x1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=16, stride=1, padding=0)
        self.upsample_2x2 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=15, stride=1, padding=0)
        self.upsample_3x3 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=14, stride=1, padding=0)
        self.upsample_6x6 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=11, stride=1, padding=0)
        self.upsample_9x9 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=8, stride=1, padding=0)

        if self.use_residuals:
            self.upsample_32x32_res = nn.ConvTranspose2d(in_channels=2688, out_channels=1024, kernel_size=3, stride=2,
                                                         padding=1, output_padding=1)
            self.upsample_64x64_res = nn.ConvTranspose2d(in_channels=1536, out_channels=512, kernel_size=3, stride=2,
                                                         padding=1,
                                                         output_padding=1)
            self.upsample_128x128_res = nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=3, stride=2,
                                                           padding=1, output_padding=1)
            self.upsample_256x256_res = nn.ConvTranspose2d(in_channels=384, out_channels=32, kernel_size=3, stride=2,
                                                           padding=1,
                                                           output_padding=1)
            self.upsample_512x512_res = nn.ConvTranspose2d(in_channels=96, out_channels=out_channels, kernel_size=4,
                                                           stride=2,
                                                           padding=1)

        else:
            self.upsample_32x32 = nn.ConvTranspose2d(in_channels=2688, out_channels=1024, kernel_size=3,
                                                     stride=2,
                                                     padding=1, output_padding=1)
            self.upsample_64x64 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,
                                                     padding=1,
                                                     output_padding=1)
            self.upsample_128x128 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                                                       stride=2,
                                                       padding=1, output_padding=1)
            self.upsample_256x256 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=2,
                                                       padding=1,
                                                       output_padding=1)
            self.upsample_512x512 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=4,
                                                       stride=2,
                                                       padding=1)

    def forward(self, x, residuals):
        x1 = self.pool1x1(x)
        x2 = self.pool2x2(x)
        x3 = self.pool3x3(x)
        x6 = self.pool6x6(x)
        x9 = self.pool9x9(x)

        x1 = self.conv1x1_1(x1)
        x2 = self.conv1x1_2(x2)
        x3 = self.conv1x1_3(x3)
        x6 = self.conv1x1_6(x6)
        x9 = self.conv1x1_9(x9)

        x1 = self.upsample_1x1(x1)
        x2 = self.upsample_2x2(x2)
        x3 = self.upsample_3x3(x3)
        x6 = self.upsample_6x6(x6)
        x9 = self.upsample_9x9(x9)

        x = torch.cat([x, x1, x2, x3, x6, x9], dim=1)

        if self.use_residuals:
            x0 = residuals[0]
            x1 = residuals[1]
            x2 = residuals[2]
            x3 = residuals[3]

            x = self.upsample_32x32_res(x)
            x = torch.sigmoid(x)

            x = torch.cat([x3, x], dim=1)
            x = self.upsample_64x64_res(x)
            x = torch.sigmoid(x)

            x = torch.cat([x2, x], dim=1)
            x = self.upsample_128x128_res(x)
            x = torch.sigmoid(x)

            x = torch.cat([x1, x], dim=1)
            x = self.upsample_256x256_res(x)
            x = torch.sigmoid(x)

            x = torch.cat([x0, x], dim=1)
            x = self.upsample_512x512_res(x)
            x = torch.sigmoid(x)
        else:
            x = self.upsample_32x32(x)
            x = torch.sigmoid(x)
            x = self.upsample_64x64(x)
            x = torch.sigmoid(x)
            x = self.upsample_128x128(x)
            x = torch.sigmoid(x)
            x = self.upsample_256x256(x)
            x = torch.sigmoid(x)
            x = self.upsample_512x512(x)
            x = torch.sigmoid(x)
        global counter
        counter = 0
        return x
