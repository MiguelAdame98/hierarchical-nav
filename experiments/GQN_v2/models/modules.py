import numpy as np
import random
import torch.nn
from dommel_library.distributions.multivariate_normal import MultivariateNormal
from dommel_library.nn import get_activation
from dommel_library.datastructs import TensorDict

class SelectContext(torch.nn.Module):
    """
    Module that randomly selects a context length, and extracts the
    relevant information from this
    """

    def __init__(self, context_size, min_context=3, random_length=False):
        torch.nn.Module.__init__(self)

        self._min_context = min_context
        self._context_size = context_size
        self._random_length = random_length

    def forward(self, *args):
        n = self._context_size
        if self._random_length:
            n = np.random.randint(self._min_context, self._context_size)
        indices = np.arange(args[0].size(1))
        np.random.shuffle(indices)
        return tuple(a[:, indices[:n]] for a in args)


class RandomSeqQuery(torch.nn.Module):
    """  create random sequences of representations and query
    """

    def __init__(self, min_context=None):
        torch.nn.Module.__init__(self)
        self.min_context=min_context

    def forward(self, sequence, set_q_idx=None):
        seq_size = sequence[list(sequence.keys())[0]].shape[1]
        min_context = self.min_context
        if set_q_idx==None or set_q_idx >= seq_size:
            if self.min_context == None or self.min_context >= seq_size:
                min_context = round(seq_size/6)
            if seq_size > 2:
                q_idx = np.random.randint(min_context,seq_size-1)
            else :
                q_idx = seq_size -1
        else:
            q_idx = set_q_idx  

        seq = TensorDict({})
        for key, value in sequence.items():
            seq[key] = value[:,:q_idx,...]
            seq[key+'_query'] = value[:,q_idx:,...]
        return seq

class RandomSelectQuery(torch.nn.Module):
    """  create random length sequences of representations and select queries randomly from anywhere on seq
    """

    def __init__(self, min_context=None, max_query=6, min_query=1):
        torch.nn.Module.__init__(self)
        self.min_context = min_context
        self.max_query = max_query
        self.min_query = min_query

    def forward(self, sequence, set_q_idx=None):
        # """  sequence: the full sequence of image and pose
        # set_q_idx: (optional) specific number of query we want
        # The sequence is divided in representation and query, with representation being a sequence of min_context starting from pose 0
        # """  
        seq_size = sequence[list(sequence.keys())[0]].shape[1]
        min_context = self.min_context
        max_query = self.max_query
        if set_q_idx==None or set_q_idx >= seq_size:
            if self.min_context == None or self.min_context >= seq_size:
                min_context = round(seq_size/6)
            if self.max_query >= seq_size: 
                max_query = round(seq_size/6)+2
            if seq_size > 2:
                #in representation seq, min amount of data: min context, max amount of data: up to the whole seq-x
                q_idx = np.random.randint(min_context,seq_size-self.min_query)
                set_q_idx = np.random.randint(self.min_query ,max_query)
                
            else :
                q_idx = seq_size -1
                set_q_idx = 1
                
        else:
            q_idx = seq_size - set_q_idx 
        #Generate set_q_idx random numbers between 1 and max_query
        randomquerylist = random.sample(range(0, seq_size), set_q_idx) 

        seq = TensorDict({})
        for key, value in sequence.items():
            seq[key] = value[:,:q_idx,...]
            seq[key+'_query'] =  torch.stack([value[:,i,...] for i in randomquerylist], dim=1)
            
        return seq

class LogDict(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, *input):
        for i in input:
            try:
                print(i.shape)
            except:
                print(i)
        return tuple(input)


class ImageDistribution(torch.nn.Module):
    def __init__(self, variance, **kwargs):
        torch.nn.Module.__init__(self)
        self._var = variance

    def forward(self, x):
        return MultivariateNormal(x, self._var * torch.ones_like(x))


class LearnableMask(torch.nn.Module):
    def __init__(self, mask_shape=(3, 64, 64)):
        torch.nn.Module.__init__(self)
        self._mask = torch.nn.Parameter(
            torch.rand(mask_shape), requires_grad=True
        )
        self._act = get_activation("Sigmoid")

    def forward(self, x):
        return x * self._act(self._mask)


class Reshape(torch.nn.Module):
    """nn.Module for the PyTorch view method"""

    def __init__(self, out_shape, **kwargs):
        """
        :param shape: New shape to reshape the data into, should be a tuple
        """
        torch.nn.Module.__init__(self)
        self._shape = out_shape

    def forward(self, x):
        return x.reshape(x.data.size(0), *self._shape)


class Sum(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, x, y):
        return x + y

import matplotlib.pyplot as plt
class VisualizeLayer(torch.nn.Module):

    def __init__(self, filename):
        torch.nn.Module.__init__(self)
        self.filename = filename
        self.calib = False

    def forward(self, x):
        if not self.calib:
            b, c, h, w = x.shape
            
            if (c > 2):
                fig, ax = plt.subplots(c//16, 16, figsize=(16 * 2, c//16 * 2))
                [a.axis('off') for a in ax.flatten()]
                if x.type() == 'torch.quantized.QUInt8Tensor':
                    x_p = torch.int_repr(x).numpy()
                    for bi in range(c//16):
                        for ci in range(16):
                            if c//16 == 1:
                                ax[ci].imshow(x_p[0, ci])
                                ax[ci].set_title(f"Ch: {ci}")
                            else:
                                ax[bi, ci].imshow(x_p[0, 16*bi + ci])
                                ax[bi, ci].set_title(f"Ch: {16*bi + ci}")
                    plt.savefig(self.filename, bbox_inches="tight")
                else:
                    for bi in range(c//16):
                        for ci in range(16):
                            if c//16 == 1:
                                ax[ci].imshow(x[0, ci])
                                ax[ci].set_title(f"Ch: {ci}")
                            else:
                                ax[bi, ci].imshow(x[0, 16*bi + ci])
                                ax[bi, ci].set_title(f"Ch: {16*bi + ci}")
                    plt.savefig(self.filename, bbox_inches="tight")
            else:
                fig, ax = plt.subplots(1, 2, figsize=(2 * 2, 1 * 2))
                [a.axis('off') for a in ax.flatten()]
                if x.type() == 'torch.quantized.QUInt8Tensor':
                    x_p = torch.int_repr(x).numpy()
                    for ci in range(2):
                        ax[ci].imshow(x_p[0, ci])
                        ax[ci].set_title(f"Ch: {ci}")
                else:
                    for ci in range(2):
                        ax[ci].imshow(x[0, ci])
                        ax[ci].set_title(f"Ch: {ci}")
                plt.savefig(self.filename, bbox_inches="tight")
            plt.show()
        return x
