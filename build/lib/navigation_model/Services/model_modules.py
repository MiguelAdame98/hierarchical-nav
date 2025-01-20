import gc
import os.path
import glob
import torch
import numpy as np
from dommel_library.modules.dommel_modules import tensor_dict

def get_model_parameters(log_dir:str, epoch:int=None)->str:
    model_dir = os.path.join(log_dir, "models")
    if epoch is None:
        model_files = glob.glob(os.path.join(model_dir, "*.pt"))
        model = max(model_files, key=os.path.getctime)
    else:
        model = os.path.join(model_dir, "model-{:04d}.pt".format(epoch))
    return model

def no_vel_no_action(sensor_data: dict) -> dict:
    """ If we don't measure a velocity, the action effectuated is Null"""
    if sum(sensor_data['vel_ob']) == 0 :
        sensor_data['action'] = np.array([0,0,0])
    return sensor_data


def torch_and_sample_observations(data: dict, observations_keys:list, sample:int, \
                    lenghts_shape_by_ob:dict = {'pose':3, 'image':4, 'action':2, 'vel_ob':3})-> dict:
    ob = torch_observations(data, observations_keys)
    ob = sample_observations(ob, observations_keys, sample, lenghts_shape_by_ob)
    return ob

def sample_observations(data:dict, observations_keys: list, sample:int, \
                    lenghts_shape_by_ob:dict = {'pose':3, 'image':4, 'action':2, 'vel_ob':3}):
    ''' sample observations with static length shape adapted to observations'''
    matching_keys = list(set(data.keys()) & set(observations_keys))
    observations = tensor_dict({}) 

    for key in matching_keys:
        length_shape = lenghts_shape_by_ob[key]
        observations[key] = sample_ob(data[key].clone(), sample, length_shape=length_shape)
        
    return observations
        
def torch_observations(sensor_data:dict, observations_keys: list) -> dict:
    ''' Input numpy data and return torch data with all possible relevant info'''
    matching_keys = list(set(sensor_data.keys()) & set(observations_keys))
    observations = tensor_dict({}) 
    for key in matching_keys:
        if 'pose' in key :
            observations[key] = torch_pose(sensor_data[key])

        if 'image' in key:
            observations[key] = torch_image(sensor_data[key])

        if 'action' in key:
            observations[key] = torch_action(sensor_data[key])
        
        if 'vel_ob' in key:
            observations[key] = torch_vel_ob(sensor_data[key])
        
    return observations

def torch_image(image) -> torch.Tensor: 
    if image is None:
        return image 
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)
    if image.shape[-1] in [1, 3]:
        #we make sure to change the shape from [..,x,y,z] to [..,z,x,y]
        image = torch.transpose(image, -1, -3)
        image = torch.transpose(image, -1, -2)
    if image.dtype == torch.uint8:
        image = image.float()
    if torch.any(image > 1):
        image /= 255
    return image

def torch_pose(pose) -> torch.Tensor:
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose).float()
    return pose

def torch_action(action) -> torch.Tensor:
    return torch.tensor(action).float()

def torch_vel_ob(vel_ob) -> torch.Tensor:
    return torch.tensor(vel_ob)
 
def sample_ob(ob:torch.Tensor, sample:int, length_shape:int = 4) -> torch.Tensor:
    while len(ob.shape)<length_shape:
        ob = ob.unsqueeze(0)
    if ob.shape[0] != 1: 
        #If image is already sampled... we squish that prior sample
        ob = torch.mean(ob, dim=0).unsqueeze(0)
    dimensions = [sample] + [1]*(length_shape-1)
    ob = ob.repeat(*dimensions)
    return ob
    
        
def delete_object_from_memory(objects):
    for object in objects:
        del object
    gc.collect()
