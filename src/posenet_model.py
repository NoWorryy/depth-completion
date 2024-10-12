import torch
from torch import nn
import networks


class PoseNetModel(nn.Module):
    '''
    Pose network for computing relative pose between a pair of images

    Arg(s):
        encoder_type : str
            posenet, resnet18, resnet34
        rotation_parameterization : str
            rotation parameterization: axis
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type='posenet',
                 rotation_parameterization='axis',
                 weight_initializer='xavier_normal',
                 activation_func='leaky_relu',
                 device=torch.device('cuda')):
        
        super(PoseNetModel, self).__init__()
        self.device = device

        # Create pose encoder
        if encoder_type == 'posenet':
            self.encoder = networks.PoseEncoder(
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256, 256, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        elif encoder_type == 'resnet18':
            self.encoder = networks.ResNetEncoder(
                n_layer=18,
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        elif encoder_type == 'resnet34':
            self.encoder = networks.ResNetEncoder(
                n_layer=34,
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))

        # Create pose decoder
        if encoder_type == 'posenet':
            self.decoder = networks.PoseDecoder(
                rotation_parameterization=rotation_parameterization,
                weight_initializer=weight_initializer,
                input_channels=256)
        elif encoder_type == 'resnet18' or encoder_type == 'resnet34':
            self.decoder = networks.PoseDecoder(
                rotation_parameterization=rotation_parameterization,
                input_channels=256,
                n_filters=[256, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))


    def forward(self, image0, image1):
        '''
        Forwards the inputs through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                image at time step 0
            image1 : torch.Tensor[float32]
                image at time step 1
        Returns:
            torch.Tensor[float32] : pose from time step 1 to 0
        '''

        # Forward through the network
        latent, _ = self.encoder(torch.cat([image0, image1], dim=1))
        output = self.decoder(latent)

        return output
