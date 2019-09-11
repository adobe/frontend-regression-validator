import torch.nn as nn


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,
                      padding=1, stride=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=3,
                      padding=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.outconv(x)
        return x
