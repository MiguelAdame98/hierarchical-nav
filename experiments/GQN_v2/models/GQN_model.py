import torch
import copy

from dommel_library.nn import summary
from dommel_library.datastructs import TensorDict
from dommel_library.distributions.multivariate_normal import MultivariateNormal
from experiments.GQN_v2.models import RandomSeqQuery,RandomSelectQuery, SelectContext, SceneEncoder as Posterior, SceneDecoder as Likelihood

from experiments.GQN_v2.models import PositionalDecoder



class GQNModel(torch.nn.Module):
    def __init__(
        self,
        SceneEncoder,
        SceneDecoder,
        observations = {'image':[3,64,64], 'pose':3},
        device = 'cpu',
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        self._observations= observations
        self._observation_shuffle = None
        
        if 'RandomSeqQuery' in kwargs and kwargs['RandomSeqQuery']==True:
            self._random_seq_query = RandomSeqQuery(kwargs.get('min_context', None))
        elif 'RandomSelectQuery' in kwargs and kwargs['RandomSelectQuery']==True:
            self._random_seq_query = RandomSelectQuery(kwargs.get('min_context', None),kwargs.get('max_query', 10), kwargs.get('min_query', 1))
        
        if 'PositionalDecoder' in kwargs:
            self._pose_decoder = PositionalDecoder(**{**kwargs.get('PositionalDecoder'), **observations, 'device': "cpu"})
            pose_encoded_dim = kwargs['PositionalDecoder'].get('pose_encoded_dim',9)
        else:
            self._pose_decoder = None
            pose_encoded_dim = 0
        
        self._posterior = Posterior(**{**SceneEncoder, **observations, 'device': device, 'pose_encoded_dim': pose_encoded_dim})
        try:
            input_length = self._posterior.output_length
        except:
            input_length = None
        self._scene_decoder = Likelihood(**{**SceneDecoder, **observations, 'input_length':input_length, 'pose_encoded_dim': pose_encoded_dim,'device': device})

        
        if 'SelectContext' in kwargs:
            self._observation_shuffle = SelectContext(**kwargs['SelectContext'])
        self._state, self._post = None,None

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state(self):
        return self._state

    def state_size(self):
        return self._num_states

    def action_size(self):
        return self._num_actions

    def observations(self):
        return [*self._observations.keys()]

    def observation_size(self, key):
        return self._observations[key]

    def reset(self, state=None, post=None):
        self._state = state
        self._post = post
        

    def fork(self, batch_size=None):
        c = copy.copy(self)
        if batch_size is None:
            c._state = self._state.clone().detach()
        else:
            if len(self._state.shape) < 2:
                c._state = (self._state.expand(batch_size, self.state_size())
                        .clone().detach())
            else:
                cont = torch.mean(self._state, dim=0)
                c._state = (cont.repeat(batch_size, 1).clone().detach())

        
        return c


    def random_seq_query(self,obs,set_q_idx=None):
        return self._random_seq_query(obs, set_q_idx)


    def __call__(self, input_dict, reset=True, **kwargs):
        if reset:
            self.reset()
        #if batch but only 1 context ob

        
        if len(input_dict.shape) == 1:
            input_dict = input_dict.unsqueeze(1)
        obs = TensorDict({key: value for key, value in input_dict.items()
                            if key != 'place'})
        place = input_dict.get('place', None)

        train = kwargs.get('train', True)
        if train==True and self._random_seq_query:
            obs = self._random_seq_query(obs)
        
        
        # if self._observation_shuffle:
        #     obs = self._observation_shuffle(obs)[0]
              
        result = self.forward(obs, place, **kwargs)

        # print('in GQN call')
        # for k,v in result.items():
        #     print(k,v.shape)

        return result
        
    def forward(self, observations, place= None, reconstruct=True, **kwargs):
        result= TensorDict({})
        result.update(observations)
        
        if place is not None :
            if type(place) == torch.Tensor:
                if place.shape[-1] == self._observations['z_size']:
                    self._post =  place
                    self._state = self._post.sample()
                else:
                    self._post =  MultivariateNormal(place[:,:,:int(place.shape[-1] / 2)],place[:,:,int(place.shape[-1] / 2):])
                    self._state = place
            else:
                mu, sigma = place._mu_sigma()
                self._post  = MultivariateNormal(mu, sigma)
                self._state = self._post.sample()
            
            
        enc_post = None
        #if we have more info than just pose
        if 'image' in observations:
            
            #if torch version >=1.10 : self._post = self._posterior(observations['image'],observations['pose'], posterior=self._post)
          
            self._post, enc_post= self._posterior(observations['image'],observations['pose'], posterior=self._post)
            #else:
            self._post= MultivariateNormal(self._post.mean.unsqueeze(1), self._post.stdev.unsqueeze(1))
            
            self._state = self._post.sample()

            #TEMPORARY TEST
            
            # enc_post_state = enc_post.sample()  
            # pose_query = observations['pose']
            # for i in range(pose_query.shape[1]):
            #     enc_post_image_predicted = self._scene_decoder(pose_query[:,i], enc_post_state[:,0])
            #     enc_post_image_predicted =enc_post_image_predicted.unsqueeze(1)
            #     if 'mid_post_image_predicted' not in result:
            #         result['mid_post_image_predicted'] = enc_post_image_predicted
            #     else:
            #         result['mid_post_image_predicted'] = torch.cat((result['mid_post_image_predicted'], enc_post_image_predicted), dim=1)

        result.update(place = self._post)
        result.update(state= self._state)
        
        #result.update(pose_encoded= pose_encoded)
        if reconstruct == True :
            if 'pose_query' in observations:
                for i in range(observations['pose_query'].shape[1]):
                    image_predicted = self._scene_decoder(observations['pose_query'][:,i], self._state[:,0])
                    image_predicted =image_predicted.unsqueeze(1)
                    if 'image_predicted' not in result:
                        result['image_predicted'] = image_predicted
                    else:
                        result['image_predicted'] = torch.cat((result['image_predicted'], image_predicted), dim=1)

            if self._pose_decoder and 'image_query' in observations:
                for i in range(observations['image_query'].shape[1]):
                    pose_predicted = self._pose_decoder(observations['image_query'][:,i], self._state[:,0])
                    pose_predicted =pose_predicted.unsqueeze(1)
                    if 'pose_predicted' not in result:
                        result['pose_predicted'] = pose_predicted
                    else:
                        result['pose_predicted'] = torch.cat((result['pose_predicted'], pose_predicted), dim=1)
                   
        return result

        
    def summary(self):
        inputs = TensorDict({})
        for key, shape in self._observations.items():
            inputs[key] = torch.zeros(shape)
            inputs[key] = inputs[key].expand(2,*inputs[key].shape)
        inputs = inputs.unsqueeze(0).to(self.device)
        
        summary(self, inputs)

    def save(self, path):
        torch.save(self.state_dict(), path)
        new_path = path[:-2] + "full.pt"
        torch.save(self, new_path)

    def load(self, path, map_location= None):
        params = torch.load(path,
                            map_location=map_location)
        self.load_state_dict(params)

    @staticmethod
    def load_full(path):
        return torch.load(path)


    # def __str__(self):
    #     summary = ''
    #     # find all NNs in the model
    #     for (k, v) in self.__dict__.items():
    #         if isinstance(v, nn.Module):
    #             summary += k + " : " + str(v) + "\n"
    #         elif isinstance(v, dict):
    #             # sometimes we also store NNs in a dict
    #             for (kk, vv) in v.items():
    #                 if isinstance(vv, nn.Module):
    #                     summary += k + "." + kk + " : " + str(vv) + "\n"

    #     return summary