# Models

## Architectures
In our experiments, we have tried models based on the encoder-decoder architecture, that contain a downsampling and an upsampling path. Currently, in the UI code, we are using `frednetv2`, but you can go ahead and change it to see the results we have achieved with the others.

In the table below, we have explained the differences between the implementations of `UNet`, `PSPNet` and `Frednetv2`, since the other `frednet` iterations are similar.

| Name        | Details         | 
| ------------- |:-------------:| 
| UNet      | Based on the implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Contains an encoder-decoder path with residual connections. It was initially used to segment images of brain scans, however we adapted it for our purpose. | 
| PSPNet      | Based on [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105). It is similar to the other encoder-decoder architectures, but the residual connections are achieved by concantenating layers of different sizes from the encoder and passing them through the decoder.      |
| FrednetV2 | This is an original architecture. We also base our model on the encoder-decoder networks, but we empirically determined the depth needed for our dataset. Another improvement came from using [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) to remove the checkerboard artifacts obtained from the learnable upsampling layers.     |
