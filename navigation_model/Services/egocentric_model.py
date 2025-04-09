import os.path
import torch
import numpy as np
from pathlib import Path
import yaml
from torch.distributions import kl_divergence
from torch.nn.functional import interpolate

from dommel_library.datastructs import cat as cat_dict
from dommel_library.nn import module_factory

from experiments.OZ.models import  ConvModel
from navigation_model.Services.base_perception_model import PerceptionModel
from navigation_model.Services.model_modules import (get_model_parameters, torch_observations, sample_observations)
from dommel_library.modules.dommel_modules import  (multivariate_distribution)

def init_egocentric_process(env_actions:list, device:str = 'cpu') -> object:
    try:
        oz_config = yaml.safe_load(Path('runs/OZ/OZ_AD_Col16_8_id/OZ.yml').read_text())
    except FileNotFoundError:
        raise ' ddddddddegocentric yaml config path file not found in runs/OZ/OZ_AD_Col16_8_id/OZ.yml.'
    if 'params_dir' in oz_config:
        oz_config['params'] = oz_config['params_dir'] 
    if 'device' in oz_config:
        oz_config['device'] = device
        return EgocentricProcess(oz_config, env_actions)
    
class EgocentricProcess():
    def __init__(self, config: dict, possible_actions: list) -> None:
        self.egocentric_model = EgocentricModel(config)
        # left = [0, 0, 1] 
        # right = [0,1,0]
        # forward = [1,0,0]
        self.possible_actions = possible_actions #[forward, right, left]

    def get_observations_keys(self) -> list:
        return self.egocentric_model.get_observations_keys()
    
    def get_egocentric_posterior(self)-> torch.Tensor:
        return self.egocentric_model.get_post()
    
    def digest(self, observations: dict, sample:int = 1) -> None:
        '''
        update egocentric model state with new data
        '''
        self.egocentric_model.digest(observations, sample=sample, reconstruct=True)

    def one_step_egomodel_prediction(self, num_samples):
    # get the list of possible actions
        list_of_action = self.possible_actions.copy()
        oz_predicted_ob = None

        #== SAVE THE EGOCENTRIC MODEL 1 STEP PREDICTIONS FOR THE POSSIBLE ACTIONS WTHOUT COLISIONS ==#
        for action in list_of_action[:]: #F/R/L
            collision_step, oz_prediction =  self.ego_model.predict(torch.tensor([action]), sample= num_samples, reconstruct=True, collision_condition=True)
            #if we have a collision 
            if collision_step == 0: 
                list_of_action.remove(action)
                continue
            if oz_predicted_ob is None:
                oz_predicted_ob = oz_prediction['image_reconstructed']
            else:
                oz_predicted_ob = torch.cat((oz_predicted_ob, oz_prediction['image_reconstructed']), dim=1)
        return list_of_action, oz_predicted_ob
    
    
    def egocentric_policy_assessment(self, policy:list, num_samples: int, prediction_wanted:bool=False):

        # Define if no collision in action_seq, and if so, truncate action seq up to there
        collision_step, oz_prediction =  self.egocentric_model.predict(policy, sample= num_samples, reconstruct=True, collision_condition=True)
        #consider the policy up to collision (collision not considered)
        policy = policy[:collision_step]
        if prediction_wanted == True:
            return policy, oz_prediction
        
        return policy

    
class EgocentricModel(PerceptionModel):

    def __init__(self, config):
        PerceptionModel.__init__(self)

        #=== Create Model structure ===#
        model_config = config['model']

        model_type = model_config.get("type", None)
        if model_type == "Conv":
            self.model = ConvModel(**model_config)
        else:
            self.model = module_factory(**model_config)
        
        #=== Load Model params ===#
        epoch = config.get('model_epoch', None)
        
        if os.path.isdir(config['params']):
            model_param = get_model_parameters(config['params'], epoch)
        else: #we consider the file as a .pt without checking
            model_param = config['params']
        try:
            self.model.load(model_param,map_location=config['device'])
        except: 
            self.model.load(model_param)
            

        self.model.to(config['device'])
        self.observations_keys = config['dataset']['keys'] 
        self.observations_shape_length = {'image':4, 'action':2, 'vel_ob':3}
     
    def get_observations_keys(self) -> list :
        ''' return ego model observations keys used as input'''
        return self.observations_keys
    
    def torch_observations(self,sensor_data: dict) -> dict:
        ''' 
        Extract from sensor_data the observations the egocentric model uses
        and adapt them for pytorch
        '''
        observations = torch_observations(sensor_data, self.observations_keys)
        if 'image' in observations:       
                observations.update(image = self.resize_image(observations['image']))
        
        return observations
    
    def sample_observations(self, observations:dict, sample:int) -> tuple[torch.Tensor,dict]:
        ''' all the observations are sampled to have the desired batch size'''
        obs = sample_observations(observations, self.observations_keys, sample, self.observations_shape_length)
        for key in obs:
            self.move_to_model_device(obs[key])   
          
            if key == 'action':
                action = obs[key] 
        del obs['action']
            
        return action, obs

    def move_to_model_device(self, tensor):
        return tensor.to(self.model.device) 
    
    def digest(self, observations:dict,  sample:int=1, reconstruct:bool= True) -> dict: 
        '''
        returns the model forward outputs containing current prior/posterior and expected surprise
        if reconstruct is True, then the output will also contains predicted obs
        '''
        #TODO: adapt for several sensors
        action, obs = self.sample_observations(observations, sample)

        with torch.no_grad():
            step = self.model.forward(action, obs, reconstruct=reconstruct)
            #print("we are inside the egocentric digest, just to see if it is updating", step)
            step.surprise = kl_divergence(step.posterior, step.prior)
            print("what kldivergence?", step.surprise)
            
           
        return step.squeeze(0)

    def predict(self, actions:torch.Tensor, sample:int=1, reconstruct:bool=False, collision_condition:bool=False) -> dict:
        '''
        predict an action sequence. params
        reconstruct: wether we want the reconstruction or not
        collision_condition: wether we consider a collision as an action sequence abortion (return the step at which there is collision)
        '''
        # returns dict with imagined actions/observations
        actions = actions.repeat(sample,1,1).to(self.model.device)
        lookahead = actions.shape[1]
        print("lookahead?",lookahead)
        fork = self.model.fork(sample)
        result = []
        # lookahead
        with torch.no_grad():
            for step in range(lookahead):
                future_step = fork.forward(
                    actions[:, step, :], reconstruct=reconstruct)
                if collision_condition == True :
                    if 'collision_reconstructed'in future_step:
                        collision = future_step['collision_reconstructed']
                        collision = int(np.round(np.mean(collision.cpu().detach().numpy())))
                    else:
                        collision = 0
                    #here we consider the collsion detection near perfect
                    if collision == 1:
                        print("we collide")
                        try:
                            print("we tried")
                            predictions = cat_dict(*result)
                        except IndexError:
                            print("we tried but failed")
                            predictions = future_step
                        return step, predictions
                result.append(future_step.unsqueeze(1))
        predictions = cat_dict(*result)
        return lookahead, predictions


    def reset(self, state:torch.Tensor) -> None:
        if state is not None:
            state = state.to(self.model.device)
        self.model.reset(state)

    def reconstruct_ob(self, state:torch.Tensor, key:str='image') -> torch.Tensor:
        ''' 
        reconstruct the observation given a state (posterior or post sample)
        the key allows you to determine which reconstruction you want.
        '''
        if isinstance(state, dict):
            state = state['dist']
        state = state.to(self.model.device)
        if type(state)== type(multivariate_distribution(torch.zeros(1),torch.ones(1))):
            state = state.sample()
        with torch.no_grad():
            return self.model.likelihood(state, key)[key]


    def fork(self, batch_size:int = None) -> object:
        return self.model.fork(batch_size)

    def get_post(self) -> torch.Tensor:
        ''' return model posterior'''
        return self.model.get_post()

    def resize_image(self, image:torch.Tensor)-> torch.Tensor:
        ''' Transforn image into model desired shape'''
        model_width = self.model.observation_size("image")[-1]
        model_height = self.model.observation_size("image")[-2]
        if model_width != image.shape[-1] or model_height != image.shape[-2]:
            image = interpolate(image, size=(model_height, model_width), mode="nearest")
    
        return image