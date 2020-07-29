import torch.nn as nn
import torch
from ml.encoder_decoder_frednetv2 import FREDv2Encoder, FREDv2Decoder
import logging

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.encoder = FREDv2Encoder()
        self.decoder = FREDv2Decoder(out_channels=5, use_residuals=True)

    def forward(self, x):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()/1024/1024/1024
            cached = torch.cuda.memory_cached()/1024/1024/1024
            reserved = torch.cuda.memory_reserved()/1024/1024/1024
            logging.debug("   .. before {:2.2f}/{:2.2f}/{:2.2f} GB of allocated/reserved/cached RAM ..".format(allocated, reserved, cached))
        x, residuals = self.encoder(x, return_residuals=True)
        #logging.debug("   .. encode {:2.2f}/{:2.2f}/{:2.2f} GB of allocated/reserved/cached RAM ..".format(allocated, reserved, cached))
        #logging.debug("   .. x is {}".format(x.size()))
        x = self.decoder(x, residuals=residuals)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()/1024/1024/1024
            cached = torch.cuda.memory_cached()/1024/1024/1024
            reserved = torch.cuda.memory_reserved()/1024/1024/1024
            logging.debug("   .. after  {:2.2f}/{:2.2f}/{:2.2f} GB of allocated/reserved/cached RAM ..".format(allocated, reserved, cached))
        return x
