from torch import nn

class ConvNet(nn.Module):
    """
    This class implements a convnet using subnet modules.
    """
    def __init__(self, encoder_config, decoder_config, classifier_config):
        super().__init__()
        
        # Init encoder 
        self.encoder = subnet(encoder_config)
        
        # Init decoder
        self.decoder = subnet(decoder_config)
        
        # Init classifier
        self.classifier = subnet(classifier_config)
    
    def forward(self, x, decode=False, classify=False):
        x = self.encoder(x)

        if decode:
            x = self.decoder(x)
        if classify:
            x = self.classifier(x)
        
        return x

class subnet(nn.Module):
    """
    This class implements subnets to be used as building pieces for more complex neural networks.
    """
    def __init__(self, subnet_config):
        super().__init__()
        self.layers = nn.ModuleList(subnet_config)
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
