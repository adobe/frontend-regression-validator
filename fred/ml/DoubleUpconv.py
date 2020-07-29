import torch.nn as nn


class DoubleUpconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleUpconv, self).__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.l2 = nn.ReflectionPad2d(1)
        self.l3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0,
                            bias=True)
        self.l4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
