
import logging
import torch
import torch.nn as nn
import numpy as np

from dommel_library.nn import get_activation
from dommel_library.nn import MLP, UpConvPipeline


from experiments.GQN_v2.models import PositionalEncoder

logger = logging.getLogger(__name__)

def no_process(x):
    return x

class MiniConvBlock(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        activation="LeakyReLU",
        residual=False,
    ):

        nn.Module.__init__(self)

        self.conv_1 = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = nn.Conv2d(
            channels_out, channels_out, kernel_size=3, stride=1, padding=1
        )
        self.activation = get_activation(activation)

        self.residual = residual

    def forward(self, x):
        # Change number of channels
        r = self.activation(self.conv_1(x))
        # process
        x = self.conv_2(r)
        # Res connect
        if self.residual:
            return x + r
        return x


class SceneDecoder(nn.Module):
    """
    In contrast to the image decoder, this decoder
    starts from a viewpose. Using a linear layer it is
    able to reconstruct
    """

    def __init__(
        self,
        channels,
        z_size,
        input_length,
        MLP_channels = [],
        activation="ReLU",
        dropout_prob=None,
        up_first=None,
        image=[3,64,64],
        pose= 3,
        ConvFilm = True,
        pose_encoded_dim = 0,
        **kwargs
    ):
        nn.Module.__init__(self)
        self._upsample_factor = 2  # (im_size / 4) ** (1 / n_layers)
        # For models with more layers, the final convolution will downsample
        self.input_length = input_length
        self.channels = channels
        
        self._activation = get_activation(
            activation, **kwargs.get("activation_args", {})
        )
        self._conv_blocks = nn.ModuleList()
        self.observations_keys = input
    
        if pose_encoded_dim > 0:
            self.pose_encoder = PositionalEncoder(pose, pose_encoded_dim)
            pose = pose_encoded_dim
        else:
            self.pose_encoder = no_process

        # conv blocks
        condition_size = pose + z_size if ConvFilm == True else 0
        # if ConvFilm == True: 
        #     condition_size = z_size + pose
        # else:
        #     condition_size = 0

        
            
        self._conv_blocks = UpConvPipeline(output_shape=image,channels=channels, condition_size=condition_size, activation=activation, flatten=False, **kwargs)

        # shape pose into an image
        if len(MLP_channels) >0 :
            self._linear_to_image = MLP(z_size + pose, np.prod(self._conv_blocks.reshape_shape), MLP_channels)
            self.input_state_pos = True
        else:
            self._linear_to_image = nn.Linear(
            pose, np.prod(self._conv_blocks.reshape_shape)
            )
            self.input_state_pos = False
        
       
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob is not None else no_process
        # if dropout_prob is not None:
        #     self._dropout = nn.Dropout2d(dropout_prob)
        # else:
        #     self._dropout = no_process

        self._upfirst = up_first
        self._im_size = image[2]
        self._compress = no_process
        self._sigmoid = get_activation("Sigmoid")

    #expect pose and state
    def forward(self, pose, state):
        pose = self.pose_encoder(pose)
        state = torch.cat([pose, state], dim=-1)

        # Learn a transformation of pose into a
        # base image
        x = self._linear_to_image(state) if self.input_state_pos == True else self._linear_to_image(pose)

        # if self.input_state_pos :
        #     x = self._linear_to_image(state)
        # else:
        #     x = self._linear_to_image(pose)
        x = self._activation(x)
        x = self._conv_blocks(x,state)
      
        return self._sigmoid(self._compress(x))


class PositionalDecoder(nn.Module):
    def __init__(self, z_size, 
                pose, image, 
                pose_encoded_dim= 9, 
                activation="ReLU", 
                channels=None, 
                device = 'cpu', 
                **kwargs):
        nn.Module.__init__(self) 
        """
        This decoder starts from a view. Using a linear layer it transform the info into pose
        """
        self._state_to_linear = MLP(np.prod(image) + z_size, pose_encoded_dim, hidden_layers= channels, activation= activation, bias=False)
        self.activation = get_activation(
            activation, **kwargs.get("activation_args", {}) )
        
        self.embed_output = nn.Linear(pose_encoded_dim, pose,  bias=False)
        self.device = device
        self.to(device)

    def forward(self, ob, state): 
        # Flatten ob
        ob = ob.reshape(-1, np.prod(ob.shape[1:]))
        #cat ob + state
        
        state = torch.cat([ob, state], dim=-1)

        x = torch.tanh(state) #-1 1
        x = self.activation(x) 
        #print('atan', x)
        x = torch.asin(x) 
        #print('arcsin x', x)
        #Reduce it to pose output
        x = self._state_to_linear(state)
        #print('self._state_to_linear',x)
        #x = self.activation(x)
        # print('activation', x)
        
        x = self.embed_output(x)
        #print('last linear to pose', x)
        #x= torch.round(x)
      
        return x

       



# =============================================================================
# Separate components needed for MultiStageDecoder
# =============================================================================
from dommel_library.nn.convolutions import Conv


class LinearToImage(nn.Module):
    def __init__(self, condition_size, channels):
        nn.Module.__init__(self)
        self._channels = channels
        self._linear_to_image = nn.Linear(
            condition_size, 4 * 4 * self._channels
        )

    def forward(self, x):
        x = self._linear_to_image(x)
        return x.reshape(-1, self._channels, 4, 4)


class FromRGB(nn.Module):
    def __init__(self, out_channels=16):
        nn.Module.__init__(self)
        self._expand = nn.Conv2d(3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self._expand(x)


class ToRGB(nn.Module):
    def __init__(self, in_channels=16):
        nn.Module.__init__(self)
        self._compress = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1)

    def forward(self, x):
        return self._compress(x)


class Resize(nn.Module):
    def __init__(self, out_shape):
        nn.Module.__init__(self)

        self._out_shape = out_shape

    def forward(self, x):
        return nn.functional.interpolate(x, self._out_shape)

