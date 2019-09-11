import torch
import torch.nn as nn
from models.encoder_decoder_frednetv4 import FREDv4Encoder, FREDv4Decoder
from models.encoder_decoder_frednetv3 import FREDv3Encoder, FREDv3Decoder
from models.encoder_decoder_frednetv2 import FREDv2Encoder, FREDv2Decoder
from models.encoder_decoder_frednet import FREDEncoder, FREDDecoder
from models.encoder_decoder_pspnet import PSPNetEncoder, PSPNetDecoder
from models.encoder_decoder_unetv2_same import UNetv2SameDecoder, UNetv2SameEncoder
from models.encoder_decoder_unet_same import UNetSameDecoder, UNetSameEncoder
from models.OutConv import OutConv

class NNet(nn.Module):
    def __init__(self, out_channels=3, use_residuals=True, model_name='frednet', out_layer=None):
        super(NNet, self).__init__()
        assert model_name is not None, 'Please choose a neural network from the available list'
        self.encoder = None
        self.decoder = None
        self.use_residuals = use_residuals
        self.use_out_layer = out_layer
        if self.use_out_layer:
            self.out_layer = OutConv(in_channels=8, out_channels=5)
        if self.use_residuals:
            print('[INFO] Residuals are set to TRUE')
        else:
            print('[INFO] Residuals are set to FALSE')
        print('[INFO] Training model: {}'.format(model_name))
        if model_name == 'frednet':
            self.encoder = FREDEncoder()
            self.decoder = FREDDecoder(out_channels=out_channels, use_residuals=use_residuals)
        elif model_name == 'unet':
            self.encoder = UNetSameEncoder()
            self.decoder = UNetSameDecoder(out_channels=out_channels, use_residuals=use_residuals)
        elif model_name == 'unetv2':
            self.encoder = UNetv2SameEncoder()
            self.decoder = UNetv2SameDecoder(out_channels=out_channels, use_residuals=use_residuals)
        elif model_name == 'pspnet':
            self.encoder = PSPNetEncoder()
            self.decoder = PSPNetDecoder(out_channels=out_channels, use_residuals=use_residuals)
        elif model_name == 'frednetv2':
            self.encoder = FREDv2Encoder()
            self.decoder = FREDv2Decoder(out_channels=out_channels, use_residuals=use_residuals)
        elif model_name == 'frednetv3':
            self.encoder = FREDv3Encoder()
            self.decoder = FREDv3Decoder(out_channels=out_channels, use_residuals=use_residuals)
        elif model_name == 'frednetv4':
            self.encoder = FREDv4Encoder()
            self.decoder = FREDv4Decoder(out_channels=out_channels, use_residuals=use_residuals)

    def forward(self, x):
        x_middle, residuals = self.encoder(x, return_residuals=True)
        if self.use_residuals:
            assert residuals is not None, 'No residuals provided'

        x_final = self.decoder(x_middle, residuals=residuals)

        if self.use_out_layer:
            x_out_layer = self.out_layer(torch.cat((x, x_final.detach()), dim=1))
            return x_out_layer, x_final
        return x_final
