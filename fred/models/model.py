import torch.nn as nn
from models.encoder_decoder_frednetv2 import FREDv2Encoder, FREDv2Decoder


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.encoder = FREDv2Encoder()
        self.decoder = FREDv2Decoder(out_channels=5, use_residuals=True)

    def forward(self, x):
        x, residuals = self.encoder(x, return_residuals=True)
        x = self.decoder(x, residuals=residuals)
        return x
