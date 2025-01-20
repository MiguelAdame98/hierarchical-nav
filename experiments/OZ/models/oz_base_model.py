import copy
import torch

from dommel_library.nn import module_factory, summary
from dommel_library.datastructs import TensorDict, cat
from dommel_library.distributions import StandardNormal


class BaseModel( torch.nn.Module):

    def __init__(
        self,
        num_states,
        num_actions,
        observations,
        prior,
        posterior,
        likelihood,
        device="cpu",
        **kwargs
    ):
        super(BaseModel, self).__init__()
        self._observations = observations
        self._num_actions = num_actions
        self._num_states = num_states

        self._prior, self._posterior, self._likelihoods = None, None, None

        # switch behaviour based on whether we get a Module or a config 
        if hasattr(prior, "update"):
            self._prior = module_factory(**prior)
            self._posterior = module_factory(**posterior)
            self._likelihoods = torch.nn.ModuleDict()
            if type(likelihood) == type(torch.nn.ModuleDict()):
                self._likelihoods = likelihood
            else:
                self._likelihoods= torch.nn.ModuleDict()
                for key in self._observations.keys():
                    if key in likelihood:
                        self._likelihoods[key] = module_factory(**likelihood[key])
                    elif 'modules' in likelihood:
                        self._likelihoods[key] = module_factory(**likelihood)
        
        else:
            self._prior = prior
            self._posterior = posterior
            self._likelihoods = self.check_moduleDict_observation(likelihood)
        
    
        self._state = None
        self._hidden = None
        self._posterior_dist = None

        self.to(device)


    @property
    def device(self):
        return next(self.parameters()).device

    def get_state(self):
        return self._state
    
    def get_post(self):
        return self._posterior_dist

    def state_size(self):
        return self._num_states

    def action_size(self):
        return self._num_actions

    def observations(self):
        return [*self._observations.keys()]

    def observation_size(self, key):
        return self._observations[key]
        
    def reset(self, state=None, hidden=None, posterior = None):
        self._state = state
        self._hidden = hidden
        self._posterior_dist = posterior
    
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

        if self._hidden is not None:
            hidden_copy = TensorDict({})
            for k, h in self._hidden.items():
                if batch_size is None:
                    hh = h.clone().detach()
                else:
                    if len(h.shape) < 2:
                        hh = h.expand(batch_size, h.shape[-1]).clone().detach()
                    else:
                      
                        hh = torch.mean(h, dim=0)
                        hh = (hh.repeat(batch_size, 1)
                                .clone().detach())
                hidden_copy[k] = hh

            c._hidden = hidden_copy

        return c

    def check_moduleDict_observation(self,input):
        if type(input) == type(torch.nn.ModuleDict()):
                output = input
        else:
            output= torch.nn.ModuleDict()
            for key in self._observations.keys():
                if key in input:
                    output[key] = input[key]
                elif 'modules' in input:
                    output[key] = input
        return output

    def prior(self, state, action):

        inputs = TensorDict({"state": state, "action": action})
        hidden = None
        inputs.update(hidden=self._hidden)
        output = self._prior(inputs)
        if "hidden" in output.keys():
            hidden = output["hidden"]
            del output["hidden"]

        return output, hidden

    def posterior(self, state, action, observations):
        inputs = TensorDict({key: value
                             for key, value in observations.items()})
        inputs.update(state=state, action=action)
        if self._hidden is not None:
            inputs.update(self._hidden)

        out = self._posterior(inputs)
        
        for key in observations.keys():
            if "features_"+key in out: 
                del out["features_"+key]
        return out

    def likelihood(self, state, key):
        inputs = TensorDict({"state": state})
        if self._hidden is not None:
            inputs.update(self._hidden)
        # discriminate between multiple likelihoods
        likelihood=self._likelihoods[key]
        out = likelihood(inputs)
        
        return out

    def __call__(self, input_dict, reset=True, **kwargs):
        if reset:
            self.reset()
        if len(input_dict.shape) == 1:
            input_dict = input_dict.unsqueeze(1)
        res = None
        for timestep in range(input_dict.shape[1]):
            step = input_dict[:, timestep]
            action = step["action"]
            obs = TensorDict({key: value for key, value in step.items()
                              if key is not action})

            reconstruct = kwargs.get('reconstruct', True)
            o = self.forward(action=action, observations=obs, reconstruct=reconstruct, **kwargs)
            o = o.unsqueeze(1)
            if res is None:
                res = o
            else:
                res = cat(res, o)
        return res

    def forward(self, action=None, observations=None, reconstruct=True, **kwargs):
        if self._state is None:
            self._state = torch.zeros(action.shape[0],
                                      self._num_states,
                                      dtype=torch.float32
                                      ).to(self.device)
      
        result, self._hidden = self.prior(self._state, action)

        if observations is None:
            if "prior" not in result.keys():
                raise Exception("Model should have a `prior` output")
            self._state = result.prior.sample()
            result.update(state=self._state)
            
        else:
            posterior_output = self.posterior(
                self._state, action, observations)
            if "posterior" not in posterior_output.keys():
                raise Exception("Model should have a `posterior` output")
            
            self._posterior_dist = posterior_output.posterior
            self._state = self._posterior_dist.sample()
            result.update(posterior_output, state=self._state)

        if reconstruct:

            for key in self._observations.keys():             
                out = self.likelihood(self._state, key)
                result.update(out)

        # remove the hidden key if we have a recurrent prior
        if "hidden" in result.keys():
            del result.hidden

        return TensorDict(result)

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

    def global_prior(self):
        return StandardNormal(self._state.shape[0], self._state.shape[1])

    def summary(self):
        inputs = TensorDict({})
        for key, shape in self._observations.items():
            inputs[key] = torch.zeros(shape)
        inputs["action"] = torch.zeros(self._num_actions)
        inputs = inputs.unsqueeze(0).unsqueeze(0).to(self.device)
        summary(self, inputs)
