class Residuals:
    def __init__(self, resnet_layer):
        resnet_layer.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features = output
