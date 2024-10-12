import torch, torchvision
from torch import nn
import log_utils, losses, networks, net_utils


EPSILON = 1e-8


class KBNetModel(nn.Module):
    '''
    Calibrated Backprojection Network class

    Arg(s):
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        min_pool_sizes_sparse_to_dense_pool : list[int]
            list of min pool kernel sizes for sparse to dense pool
        max_pool_sizes_sparse_to_dense_pool : list[int]
            list of max pool kernel sizes for sparse to dense pool
        n_convolution_sparse_to_dense_pool : int
            number of layers to learn trade off between kernel sizes and near and far structures
        n_filter_sparse_to_dense_pool : int
            number of filters to use in each convolution in sparse to dense pool
        n_filters_encoder_image : list[int]
            number of filters to use in each block of image encoder
        n_filters_encoder_depth : list[int]
            number of filters to use in each block of depth encoder
        resolutions_backprojection : list[int]
            list of resolutions to apply calibrated backprojection
        n_filters_decoder : list[int]
            number of filters to use in each block of depth decoder
        deconv_type : str
            deconvolution types: transpose, up
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        min_predict_depth : float
            minimum predicted depth
        max_predict_depth : float
            maximum predicted depth
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 input_channels_image, input_channels_depth,
                 min_pool_sizes_sparse_to_dense_pool, max_pool_sizes_sparse_to_dense_pool, n_convolution_sparse_to_dense_pool, n_filter_sparse_to_dense_pool,
                 n_filters_encoder_image, n_filters_encoder_depth, n_filters_decoder,
                 resolutions_backprojection,
                 deconv_type='up',
                 weight_initializer='xavier_normal',
                 activation_func='leaky_relu',
                 min_predict_depth=1.5, max_predict_depth=100.0,
                 device=torch.device('cuda')):

        super(KBNetModel, self).__init__()
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        self.device = device

        # Build sparse to dense pooling
        self.sparse_to_dense_pool = networks.SparseToDensePool(
            input_channels=input_channels_depth,
            min_pool_sizes=min_pool_sizes_sparse_to_dense_pool,
            max_pool_sizes=max_pool_sizes_sparse_to_dense_pool,
            n_convolution=n_convolution_sparse_to_dense_pool,
            n_filter=n_filter_sparse_to_dense_pool,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        # Set up number of input and skip channels
        input_channels_depth = n_filter_sparse_to_dense_pool

        n_filters_encoder = [
            i + z
            for i, z in zip(n_filters_encoder_image, n_filters_encoder_depth)
        ]

        n_skips = n_filters_encoder[:-1]
        n_skips = n_skips[::-1] + [0]

        n_convolutions_encoder_image = [1, 1, 1, 1, 1]
        n_convolutions_encoder_depth = [1, 1, 1, 1, 1]
        n_convolutions_encoder_fused = [1, 1, 1, 1, 1]

        n_filters_encoder_fused = n_filters_encoder_image.copy()

        # Build depth completion network
        self.encoder = networks.KBNetEncoder(
            input_channels_image=input_channels_image,
            input_channels_depth=input_channels_depth,
            n_filters_image=n_filters_encoder_image,
            n_filters_depth=n_filters_encoder_depth,
            n_filters_fused=n_filters_encoder_fused,
            n_convolutions_image=n_convolutions_encoder_image,
            n_convolutions_depth=n_convolutions_encoder_depth,
            n_convolutions_fused=n_convolutions_encoder_fused,
            resolutions_backprojection=resolutions_backprojection,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.decoder = networks.MultiScaleDecoder(
            input_channels=n_filters_encoder[-1],
            output_channels=1,
            n_resolution=1,
            n_filters=n_filters_decoder,
            n_skips=n_skips,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            output_func='linear',
            use_batch_norm=False,
            deconv_type=deconv_type)


    def forward(self,
                image,
                sparse_depth,
                validity_map_depth,
                intrinsics):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse depth
            validity_map_depth : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output dense depth
        '''

        # Depth inputs to network:
        # (1) raw sparse depth, (2) filtered validity map
        input_depth = [
            sparse_depth,
            validity_map_depth
        ]

        input_depth = torch.cat(input_depth, dim=1)

        input_depth = self.sparse_to_dense_pool(input_depth)

        # Forward through the network
        shape = input_depth.shape[-2:]
        latent, skips = self.encoder(image, input_depth, intrinsics)

        output = self.decoder(latent, skips, shape)[-1]

        output_depth = torch.sigmoid(output)

        output_depth = self.min_predict_depth / (output_depth + self.min_predict_depth / self.max_predict_depth)

        return output_depth