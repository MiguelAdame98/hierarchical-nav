import logging

import numpy as np
import torch
import torch.nn as nn
from dommel_library.distributions.multivariate_normal import MultivariateNormal
from dommel_library.nn import ConvFiLM, ConvPipeline, VariationalMLP, get_activation

logger = logging.getLogger(__name__)

def no_process(x):
    return x

class PositionalEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        nn.Module.__init__(self) 
        self.embed_input = nn.Linear(input_size, embedding_size, bias=False)
    
    def forward(self, x): 
        return torch.sin(self.embed_input(x))

class SceneEncoder(nn.Module):
    def __init__(
        self,
        channels,
        MLP_channels = None,
        z_size=256,
        expand=False,
        activation="ReLU",
        batch_norm=False,
        aggregation_method="kalman",
        aggregate_factor=2,
        clip_variance=0,
        dropout_prob=None,
        device="cpu",
        image=[3,64,64],
        pose = 3,
        ConvFilm= True,
        pose_encoded_dim = 0,
        **kwargs
    ):

        nn.Module.__init__(self)
        self.observations_keys = input
        self._aggregation_method = aggregation_method
        self._aggregate_factor = aggregate_factor

        self._batch_norm = batch_norm

        self.image= image.copy()
          
        if pose_encoded_dim > 0:
            self.pose_encoder = PositionalEncoder(pose, pose_encoded_dim)
            pose = pose_encoded_dim
        else:
            self.pose_encoder = no_process

        if expand:
            self._expand = nn.Conv2d(
                self.image[0], channels[0], kernel_size=1, stride=1
            )
            self.image[0] = channels[0]
            self.image[1] = int(np.ceil(image[1] / 1))
            self.image[2] = int(np.ceil(image[2] / 1))

            channels.pop(0)
        else:
           self._expand = no_process
        
        condition_size = pose if ConvFilm == True else 0
        # if ConvFilm != True: 
        #     condition_size = 0
        # else:
        #     condition_size = pose
            


        self._convs= ConvPipeline(input_shape=self.image,channels=channels, condition_size=condition_size, activation=activation, batch_norm=batch_norm, flatten=False, **kwargs)

        self.output_length= self._convs.output_length
        self.variational = VariationalMLP(
            self.output_length, z_size, MLP_channels, activation=activation
        )

        self.activation = get_activation(activation)

        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob is not None else no_process
        # if dropout_prob is not None:
        #     self.dropout = nn.Dropout2d(p=dropout_prob)
        # else:
        #     self.dropout = no_process

        self.device = device
        self.to(device)

        self._clip_variance = clip_variance

    

    def encode_single(self, x, p=None):
        """
        :param x: image
        :param p: pose. If the pose is abscent, the FiLM layer is not used
        :return: Gaussian distribution over the latent vector
        """
        p = self.pose_encoder(p)
        x = self._expand(x)
        
        x = self._convs(x, p) if p is not None else x
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        return self.variational(x)

    def forward(self, cx, cp, posterior= None):
        """
        Implementation to compute the latent vector for all the context
        information in a batched manner
        :param cx: context views
        :param cp: context poses
        :param posterior: not None if we improve a previous latent place
        :return: tensor of means (mus), tensor of standard deviations (sigmas)
        """
        #TODOD: REDO IMP
  
    
        # Receives information of the shape
        # (batch_size, context_size, channels, im_width, im_height)
        # Transform to
        # (batch_size * context_size, channels, im_width, im_height)
        x_shape, p_shape = cx.shape, cp.shape

         # batch, ...  (channels, w, h) or other dims
        cx = cx.reshape(x_shape[0]*x_shape[1],*x_shape[2:])
        # batch, .... as poses
        cp = cp.reshape(p_shape[0]*p_shape[1],*p_shape[2:])

        post = self.encode_single(cx, cp)
        
        post = post.reshape(x_shape[0], x_shape[1], -1)
        aggregated_posterior = self.aggregate_method(post, posterior)

        return aggregated_posterior, post

    def aggregate_method(self, post, posterior=None):
        #if self._aggregation_method == "kalman":
        return self.aggregate(
            post, self._aggregate_factor, self._clip_variance, posterior
        )


    @staticmethod
    def aggregate(post, aggregate_factor=1, clip_variance=0, posterior = None):
        """
        Aggregate the posterior using hierarchical multivariate multiplication
        :param posterior: shape [B, L, MultivariateNormal]
        :return:
        """
        _, c, _ = post.shape
        
        if posterior != None:
            aggregated_posterior = posterior.squeeze(1)
            start_seq = 0
            
        else:
            aggregated_posterior = MultivariateNormal(post[:, 0, :])
            start_seq = 1

        #else the for is applied
        for i in range(start_seq, c):    
            
            aggregated_posterior = MultivariateNormal(
                aggregated_posterior.mean,
                aggregated_posterior.stdev * np.sqrt(aggregate_factor),
            )
            
            # multivariate gaussian multiplication!
            aggregated_posterior = aggregated_posterior * MultivariateNormal(
                post[:, i, :]
            )
            
            # Numerical stability clip variance
            aggregated_posterior_var = torch.max(
                aggregated_posterior.variance,
                clip_variance * torch.ones_like(aggregated_posterior.variance),
            )
            aggregated_posterior = MultivariateNormal(
                aggregated_posterior.mean, aggregated_posterior_var
            )

            
        #print('aggregated_posterior last step mean: ' + str(round(torch.mean(aggregated_posterior.mean).cpu().detach().numpy().tolist(),4)) +', std: ' +str(round(torch.mean(aggregated_posterior.stdev).cpu().detach().numpy().tolist(),4))\
             #+ ' var:'+ str(round(torch.mean(aggregated_posterior.variance).cpu().detach().numpy().tolist(),4)))
     
        return aggregated_posterior

    def combine_information(self, batch_mu, batch_sig):
        """
        Receives distributions as (Batch size, context_idx, mean/var)
        :param batch_mu: batch of means
        :param batch_sig: batch of variances
        :return: distributions (batch, mean/var)
        """
        _, c, _, = batch_mu.shape
        mu, sig = batch_mu[:, 0, :], batch_sig[:, 0, :]
        for i in range(1, c):
            mu, sig = self.add_information(
                (mu, sig), (batch_mu[:, i, :], batch_sig[:, i, :])
            )
            if mu.sum() != mu.sum():
                logger.info("nan detected in combine_information")
                logger.info(batch_mu[:, i, :])
                input()
            if sig.sum() != sig.sum():
                logger.info("nan detected in combine_information")
                logger.info(batch_sig[:, i, :])
                input()
        if self._aggregation_method == "mean":
            mu *= 1 / c

        return mu, sig

    def add_information(self, d0, d1):
        """
        combine two distributions
        :param d0: distribution 0 (mu, sig)
        :param d1: distribution 1 (mu, sig)
        :return: combined distribution mu, sig
        """
        if self._aggregation_method == "kalman":
            k = d1[1] / (d0[1] + d1[1])
            mu = k * d0[0] + (1 - k) * d1[0]
            sig = (1 - k) * d1[1]
        # elif self._aggregation_method == "kalman-fixed-noise":
        #     # Scale variance by multiplying by 2. To mitigate the reduced variance
        #     # Noise model is estimated by increasing variance over observations...
        #     scaling_factor = 2
        #     k = d1[1] / (scaling_factor * d0[1] + d1[1])
        #     mu = k * d0[0] + (1 - k) * d1[0]
        #     sig = (1 - k) * d1[1]

        # elif (
        #     self._aggregation_method == "addition"
        #     or self._aggregation_method == "mean"
        # ):
        #     mu = d0[0] + d1[0]
        #     sig = torch.ones_like(mu)
        # elif self._aggregation_method == "multiplication":
        #     mu = d0[0] * d1[0]
        #     sig = torch.ones_like(mu)
        # elif self._aggregation_method == "feature_wise_max_pool":
        #     mu = torch.max(d0[0], d1[0])
        #     sig = torch.ones_like(mu)

        return mu, sig + 1e-8

