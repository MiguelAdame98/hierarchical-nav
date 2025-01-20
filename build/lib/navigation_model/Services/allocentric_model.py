import os.path
import sys
from itertools import product
from operator import itemgetter
from typing import Tuple

import numpy as np
import torch

from env_specifics.env_calls import call_env_place_range
from experiments.GQN_v2.models import GQNModel
from navigation_model.Services.model_modules import delete_object_from_memory
from navigation_model.Processes.AIF_modules import (
    calculate_FE_EFE, compute_std_dist,
    mse_observation)
from navigation_model.Processes.motion_path_modules import action_to_pose
from navigation_model.Services.base_perception_model import PerceptionModel
from dommel_library.modules.dommel_modules import (
    multivariate_distribution, tensor_dict)
from navigation_model.Services.model_modules import (get_model_parameters,
                                                     sample_observations,
                                                     torch_observations,
                                                     torch_pose)
# from navigation_model.visualisation_tools import visualise_image


def init_allocentric_process(allo_model_config:dict, env:str, device:str = 'cpu'):
    allo_model_config['device'] = device
    
    std_place_th = allo_model_config['model']['SceneEncoder']['clip_variance'] #0.25
    allocentric_process = AllocentricProcess(allo_model_config, env)
    allocentric_process.reset()
    
    return allocentric_process, std_place_th

class AllocentricProcess():
    def __init__(self, config:dict, env:str) -> None:
        self.allocentric_model = AllocentricModel(config)
        self.env_specific = env
        default_content = {'post': None, 'pose': np.array([0,0,0]), 'hypothesis_weight': 0.5, 'std': 1, 'info_gain': 0, 'exp':-1 }
        self.place_descriptors = { 0: default_content} #, 1: default_content }

        self.prev_step_place_std = sys.maxsize
        self.step_lost_counter = 0
        self.mse_threhsold = 0.5
    
    def reset(self) -> None:
        self.allocentric_model.reset()

#================= GET METHODS ==============================================================================#
   
    def get_mse_threshold(self) -> float:
        return self.mse_threhsold
    
    def get_best_place_hypothesis(self) -> dict:
        return self.place_descriptors[0]
    
    def get_all_place_hypothesis(self) -> dict:
        return self.place_descriptors
    
    def get_observations_keys(self) -> list:
        return self.allocentric_model.get_observations_keys()
    
    def get_place_range(self) -> list:
        return call_env_place_range(self.env_specific)
         
    def extract_mse_from_hypothesis(self, default_value= np.nan) -> list:
        return [item.get('mse', default_value) for item in self.place_descriptors.values()]
    
    def extract_weight_from_hypothesis(self, default_value= np.nan) -> list:
        return [item.get('hypothesis_weight', default_value) for item in self.place_descriptors.values()]
    
    def place_doubt_step_count(self) -> int:
        '''
        get how many steps the model has been hesitating on the best 
        place description 
        '''
        return self.step_lost_counter
    
    def get_place_descriptions(self) -> dict:
        ''' get all place descriptions hypothesis'''
        return self.place_descriptors
    
    def get_best_place_description(self) -> dict:
        ''' get the place description having the highest probability'''
        return self.place_descriptors[0]
    
    def get_sampled_data(self,data_dict:dict, num_samples:int):
        ''' input a TensorDict or dict of observations 
         returns it with the batch sample added'''
        return self.allocentric_model.sample_observations(data_dict, sample = num_samples)

    def confident_about_place_description(self) -> bool:
        ''' Does the place distribution have a small standard deviation?'''
        place_std = self.get_best_place_hypothesis()['std']
        # if we have a stable place modelised (std < th and not moving anymore) --> we have a strong belief
        return round(self.prev_step_place_std - place_std, 4) == 0

#=================  FORMAT METHODS ==========================================================================
    def reset_step_lost_counter(self):
        self.step_lost_counter = 0
    
    def lost_steps_increase(self) -> None:
        '''
        increase lost steps count, only if we are already lost
        '''
        if self.step_lost_counter > 0:
            self.step_lost_counter + 1

    def sample_observations(self, observations:dict, num_samples:int)-> dict:
        return self.allocentric_model.sample_observations(observations, sample = num_samples)
    
    def place_as_sampled_Normal_dist(self, place, num_samples:int)-> dict:
        ''' transform a place in a distribution and sample it '''

        #print(' place as sampled Normal dist shape and type', place.shape, type(place))
        if not isinstance(place, torch.Tensor):
            place = multivariate_distribution(torch.from_numpy(place))
        place = place.unsqueeze(0).unsqueeze(0).repeat(num_samples,1,1) 
        return place

    def create_pose_observations(self,poses:list, num_samples:int)-> dict:
        ''' torch and sample poses and save them as dict observations '''
        pose_options_dict = {'pose': list(poses)}
        pose_observations = self.allocentric_model.torch_observations(pose_options_dict)
        pose_observations = self.allocentric_model.sample_observations(pose_observations, sample = num_samples)
        return pose_observations

    def one_action_ob_update(self,observations:dict, prev_pose:list) -> Tuple[dict,list] :
        ''' update observations given hypo current pose and action'''
        pose  = action_to_pose(observations['action'],prev_pose)
        observations['pose'] = torch_pose(pose.copy())
        # observations = self.allocentric_model.sample_observations(observations,sample)
        return observations, pose  

#================= STEP UPDATE METHODS START==============================================================================#

    def update_place_believes(self, observations: dict, sample:int) -> None:
        """  
        for all models in dict, we update them and predict their mse/kl, hypothesis_weight and predicted_img
        """
        #step = [None]*len(self.place_descriptors)
        mse_list = []
        
        print('place_descriptors length', len(self.place_descriptors))
        print('pose of best hypothesis before action ', self.place_descriptors[0]['pose'])
        #Update the step count for the parallel models run
        self.lost_steps_increase()

        self.prev_step_place_std = self.get_best_place_hypothesis()['std']
        
        #copy nested dictionary to be able to simultaneously:
        # - access and modify directly place_descriptors  (it's a copy, not a deepcopy)
        # - add dicts to place_descriptors 
        tmp_model_properties = self.place_descriptors.copy()
        for place_idx, place_description in tmp_model_properties.items():
            
            #---- 1. Observations incorporation ----#
            observations, place_description['pose'] = self.one_action_ob_update(observations, place_description['pose'])
            
            #---- 2. Update Believes of the generative models ----#
            self.allocentric_model.reset() #safety net to be sure to start anew
            step = self.allocentric_model.digest(observations, place_description['post'],sample)

            #---- 3. Process beleieves and update the model properties----#
            
            #If we have a prior to our place
            place_description['mse'], place_description['kl'], place_description['image_predicted'] = self.calculate_hypo_likelihood(place_description, step)
            mse_list.append(place_description['mse'])                          
            #print('place ',place_idx, 'pose',  place_description['pose'], 'mse and kl:', place_description['mse'], place_description['kl']) 
  
            #if we have no prior for this place  OR  If our place explain our image
            if np.isnan(place_description['mse']) \
                or place_description['mse'] < self.mse_threhsold : 
                place_description = self.increase_hypo_likelihood(place_description, step)

            
            #We don't want to create thousands hypo, let's restrict the parallel creation on the best looking positions
            #If we already have an mse < th, then we know it's not worth creating new models since all will be erased. 
            #If the mse is too high and there is a lot of models, don't consider this hypo at all
            elif not min(mse_list) < self.mse_threhsold and \
                (place_description['mse'] <= 1.6 * self.mse_threhsold \
                or len(self.place_descriptors) < len(self.get_place_range())):
                place_description = self.decrease_hypo_likelihood(place_description)
                print("CREATING NEW ALTERNATIVE place")
                self.add_new_place_hypothesis(step['place'],place_idx,observations['image'], sample)
        
        if self.place_doubt_step_count() > 0 :
            self.prev_step_place_std = sys.maxsize
        return
    
    def calculate_hypo_likelihood(self, place_description:dict, step_update:dict)->Tuple[float,float, torch.Tensor]:
        if place_description['post'] != None :
            kl ,mse, _, img_pred = \
            calculate_FE_EFE(self.allocentric_model.model.fork(), step_update['place'], step_update, place_description['post'])
        else:
            mse = np.nan
            kl = np.nan
            img_pred = None
        return mse, kl, img_pred
    
    def increase_hypo_likelihood(self,place_description:dict, step_update:dict) ->dict:
        new_place = step_update['place']
        place_description['post'] = new_place
        place_description['hypothesis_weight'] += 0.5
        place_description['std'] = compute_std_dist(new_place)
        if not np.isnan(place_description['kl']):
            place_description['info_gain'] += place_description['kl']
        return place_description
    
    def decrease_hypo_likelihood(self,place_description:dict) -> dict:
        
        #hypothesis not likely anymore
        place_description['hypothesis_weight'] = 0
        if self.place_doubt_step_count() == 0: #If we just started doubting our position
            place_description['exp'] = -1
        
        #we don't keep current hypo post
        self.allocentric_model.reset()
        return place_description
    
    def lowest_mse_hypo_increase_likelihood(self, mse:list) -> int:
        """
        The hypothesis having the lowest MSE will have an 
        increased probability of being correct. If all mse are nan
        then hypothesis 0 will automatically get the precedence
        """
        best_mse_idx = self.best_model_according_mse(mse) 
        print('number of parallel options running: ', len(mse) ,' and lowest values of mse:', mse[best_mse_idx])#, kl[index_min_err])
        print('place explaining observation best conssidering the recon err:' +str(best_mse_idx))#+ ', considering kl:'+ str(index_min_kl) )
        self.add_hypo_weight_mse_under_threshold(mse, best_mse_idx)
        return best_mse_idx
    
    def assess_believes(self) -> None:
        """ Using MSE and Weight confirmation we select the model that seems to best describe the environment """
        
        #---- 1. Search for the best matching MSE and add weight to it if it's a plausible place descriptor----#
        mse = self.extract_mse_from_hypothesis(default_value= np.nan)
        self.lowest_mse_hypo_increase_likelihood(mse)
        
        #---- 2. Search for the best place descriptor having a sufficient weight to be plausible ----#
        self.highest_weight_hypo_convergence()        
        print('agent best hypothesis lp pose:', self.place_descriptors[0]['pose'])

    def best_model_according_mse(self, mse:list) -> int:
        ''' extract all mse from the place descriptions and return it with the idx of the lowest mse'''
        try:
            best_mse_idx = np.nanargmin(mse)
            #index_min_kl = np.nanargmin(kl)
        except ValueError: #all NAN
            print('value error mse', mse)
            best_mse_idx = 0 #index_min_kl = 0,0
        return best_mse_idx
    
    def add_hypo_weight_mse_under_threshold(self, mse:list, best_mse_idx:int) -> None:
        ''' if the mse is under threshold, hypo proba gain +1'''
        if mse[best_mse_idx] < self.mse_threhsold:
            print('according to MSE:')
            print('place ', best_mse_idx ,' is the best place to explain observations')
            print('pose:', self.place_descriptors[best_mse_idx]['pose'])
            #best model preferably always takes index 0
            if self.place_doubt_step_count() > 0 :
                temp = self.place_descriptors[0].copy()
                self.place_descriptors[0] = self.place_descriptors[best_mse_idx]
                self.place_descriptors[best_mse_idx] = temp
            self.place_descriptors[0]['hypothesis_weight'] += 1 #add more weight to this option  
 
    def best_model_according_to_weights(self, hypothesis_weight:list, index_max_weight:int)->int:
        """
        check if we have several hypothesis with equivalent weight, 
        return the index one having the highest weight and lowest mse 
        """
        mse = self.extract_mse_from_hypothesis(default_value= np.nan)
        list_indexes_max_weight = [idx for idx,val in enumerate(hypothesis_weight) if val==hypothesis_weight[index_max_weight]]
        print('highest weight models indexes', list_indexes_max_weight)
        best_model_idx = index_max_weight
        for idx in list_indexes_max_weight:
            if idx == 0:
                best_model_idx = idx
                break
            if mse[idx] < mse[best_model_idx]: 
                best_model_idx = idx   
            
        print('according to model weigths:')
        print('place ' + str(best_model_idx) + ' is the best place to explain observations')
        print('pose:', self.place_descriptors[best_model_idx]['pose'], 'mse:', mse[best_model_idx], 'weight:', self.place_descriptors[best_model_idx]['hypothesis_weight'])
        return best_model_idx
    
    def converge_to_one_hypothesis(self, best_model_idx:int)-> None:
        ''' The best hypothesis is considered enough to describe the environment'''
        tmp_best_hypothesis = self.place_descriptors[best_model_idx]
        prev_exp = self.place_descriptors[best_model_idx].get('exp', -1)
            
        if len(self.place_descriptors)> 1 :
            #condition added for computational memory sake
            print('we select best matching model and delete the other parallel models')
            delete_object_from_memory([self.place_descriptors])
        self.place_descriptors = { 0: {**tmp_best_hypothesis}}
        self.place_descriptors[0]['exp'] = prev_exp
    
#================= MODEL LOST MULTIPLES HYPOTHESES PROCESS ==============================================================================#
    def remove_worst_hypothesis(self, portion:float = 3 )->None:
        """
        remove a portion of the hypothesis having the worst mse. 
        The ones having none are not considered.
        """
        mse = self.extract_mse_from_hypothesis(default_value= 0)
        
        n_remove = len(mse) // portion # Calculate number of models to remove
        print('we erase the last ', n_remove, ' worst models')
        
        # Sort indices by mse, with their original index
        mse_sorted = sorted(enumerate(mse), key=lambda x: x[1], reverse=True)[:n_remove] 
        for idx, _ in mse_sorted:
            del self.place_descriptors[idx] # Remove models with highest mse
        self.place_descriptors = {idx: model_props.copy() for idx, model_props in enumerate(self.place_descriptors.values())} # Update keys
    
    def highest_weight_hypo_convergence(self)-> None:
        """
        We check the weight proba of the hypothesis, if one is above the threshold 
        we converge to the hypotheses having this weight and the lowest mse
        we erase all other hypothesis
        else, every 2 steps we erase a set portion of the worst hypothesis
        """
        hypothesis_weight = self.extract_weight_from_hypothesis(default_value=0)
        index_max_weight = np.nanargmax(hypothesis_weight)
        print('highest weight index and its value in list and dict',  index_max_weight, self.place_descriptors[index_max_weight]['hypothesis_weight'])

        #if we have likely hypo, we converge to the best hypo among them  
        if hypothesis_weight[index_max_weight] >= 2.5:
            best_model_idx = self.best_model_according_to_weights(hypothesis_weight, index_max_weight)
            self.converge_to_one_hypothesis(best_model_idx)
            self.reset_step_lost_counter()
        
        #Every 2 steps, we erase a part of the worst hypothesis
        elif self.place_doubt_step_count() >= 2:
            self.remove_worst_hypothesis(portion = 3)
    
    def add_new_place_hypothesis(self, current_updated_post, place_idx: int, current_ob: torch.Tensor, num_samples:int) -> None:
        """ we create parallel hypothesis with poses encompasing -most- possible agent position """
        #If we are here, we are doubting our belief.
        if self.step_lost_counter == 0:
            self.step_lost_counter += 1
        
        #We re-add the current wrong hypo place in the loop with the previous pose.
        self.add_parallel_hypothesis(pose=self.place_descriptors[place_idx]['pose'].copy(), post=current_updated_post[:])
        
        pose_options = self.get_place_range()
        pose_options_dict = {'pose': pose_options}
        observations = self.allocentric_model.torch_observations(pose_options_dict).squeeze(0)
           
        #we only create post:Normal hypothesis around all ranged poses given current ob 
        #when we start doubting or have been wondering for long enough to have erased enough worst hypothesis
        if len(self.place_descriptors) < len(pose_options): 
            ob = {'image': current_ob.clone()}
            for pose_idx in range(observations['pose'].shape[0]):
                pose = observations['pose'][pose_idx,...].unsqueeze(0)
                ob.update(pose= pose) 
                
                alternative_step = self.allocentric_model.digest(ob, None, sample=num_samples)
                if 'place' in alternative_step:
                    alt_post = alternative_step.place
                else:
                    alt_post = alternative_step.posterior

                self.add_parallel_hypothesis(pose= np.array(pose_options[pose_idx]).copy(), post=alt_post)
        
    def add_parallel_hypothesis(self, pose:list, post=None) -> None:
        """ We create new model property batch at the end of current dict """
        new_place_idx = len(self.place_descriptors)
        self.place_descriptors[new_place_idx] = {}
        self.place_descriptors[new_place_idx]['post'] = post
        self.place_descriptors[new_place_idx]['pose'] = pose
        self.place_descriptors[new_place_idx]['hypothesis_weight']= 0
        self.place_descriptors[new_place_idx]['std']= 1
        self.place_descriptors[new_place_idx]['info_gain']= 0
        self.place_descriptors[new_place_idx]['exp']= -len(self.place_descriptors)
        #Those do not need to be init 
        self.place_descriptors[new_place_idx]['image_predicted'] = None
        self.place_descriptors[new_place_idx]['kl'] = np.nan
        self.place_descriptors[new_place_idx]['mse'] = np.nan
        #print('new para model', self.place_descriptors[new_place_idx])
    
    def add_hypothesis_to_competition(self,place, poses, mse=None, exp=None):
        """
        we add the given place and its poses to the hypothesis
        """
        for p_idx in range(len(poses)):
            self.add_parallel_hypothesis(np.array(poses[p_idx]), post=place)
            if mse[p_idx] is None:
                #If the place do not have an associated MSE
                #Then we add the starting weight of a promising hypothesis
                weight = self.place_descriptors[len(self.place_descriptors)-1]['hypothesis_weight'] + 0.5
                self.place_descriptors[len(self.place_descriptors)-1]['mse'] = np.nan
            else:
                #If the place has an associated MSE
                #Then we add the starting weight corresponding to the MSE, with the weight
                #ranging necessarily between 0.1 and 0.55 (the min is there to avoid any sad 0)
                try:
                    weight = np.max([np.min([np.log(7*(self.mse_threhsold-mse[p_idx])),0.1]), 0.55])
                except RuntimeWarning:
                    weight = 0.1

                self.place_descriptors[len(self.place_descriptors)-1]['mse'] = mse[p_idx]
            self.place_descriptors[len(self.place_descriptors)-1]['hypothesis_weight'] = weight
                       
            if isinstance(exp, int):
                self.place_descriptors[len(self.place_descriptors)-1]['exp'] = exp

#================= STEP UPDATE METHODS ==============================================================================#

    def update_place(self, observations: dict, view_cell_place) -> None:

        print('The experience view do not match the current place')
        view_cell_place = view_cell_place.unsqueeze(0).unsqueeze(0).repeat(self.num_samples,1,1)
        pose_options = self.get_place_range()
        pose_options_dict = {'pose': pose_options}
        pose_observations = self.allocentric_model.torch_observations(pose_options_dict,self.get_observations_keys()).squeeze(0)
        pose_observations = self.allocentric_model.sample_observations(pose_observations, sample = self.num_samples)
        if 'pose' in observations:
            del observations['pose']
        find_pose_observations = self.allocentric_model.sample_observations(observations, sample = self.num_samples)
        
        mse_list = []
        for pose_idx in range(pose_observations['pose'].shape[1]):
            pose = pose_observations['pose'][:,pose_idx,...].unsqueeze(1)
            find_pose_observations.update(pose= pose) 
            #alternative_step = self.allocentric_model.digest(find_pose_observations, view_cell_place, sample=self.num_samples)
        
            mse_pose, image_predicted = mse_observation(self.allocentric_model.fork(), view_cell_place, find_pose_observations)
            mse_list.append([mse_pose, pose_idx])
        best_pose_options = sorted(mse_list, key=itemgetter(0))[:5]
        
        print('show me the 5 best pose options')
        for (mse, idx) in best_pose_options:
            print('mse, idx and pose',mse, idx, pose_options[idx] )
        
        self.place_descriptors[0]['post'] = view_cell_place
        self.place_descriptors[0]['pose'] = np.array(pose_options[best_pose_options[0][1]])

    def reset_parallel_models(self, idx: int= None) -> None:
        """ We erase selected model property based on its id """
        if idx == None:
            memory_saved = self.place_descriptors[0]
            delete_object_from_memory([self.place_descriptors])
            self.place_descriptors = {0: {**memory_saved}}
        else:
            del self.place_descriptors[idx]
            print('reset models, check model properties keys before erasing', self.place_descriptors.keys())
            for i in range (idx+1, len(self.place_descriptors)):
                self.place_descriptors[i] = self.place_descriptors[i-1]
            print('reset models, check model properties keys AFTER erasing', self.place_descriptors.keys())
    
    def allocentric_pose_prediction(self, pose: np.ndarray, place , num_samples:int) -> dict:
        ''' Given a pose and a place, what does the model predicts'''
        self.allocentric_model.reset() #safety to be sure to start anew
        pose_query = self.allocentric_model.torch_observations({'pose': list(pose)})
        pose_query = self.sample_observations(pose_query, num_samples)
        pose_query = self.allocentric_model.create_query(pose_query, keys=['pose'])
        
        predicted_step = self.allocentric_model.predict(place, pose_query, sample=num_samples)
        return predicted_step
    
    def assess_poses_plausibility_in_place(self, place:torch.Tensor, observations:dict, pose_observations:dict)-> tuple[list, list]:
        """
        for all the poses we check their predicted image mse compared to ob, 
        and we keep only the poses and mse under set threshold as valid. 
        """
        mse_pose_list = self.predicted_mse_of_poses(place, observations, pose_observations)
        mse_pose_list = sorted(mse_pose_list, key=itemgetter(0))
        selected_poses, selected_poses_mse = [], []
  
        for (mse,pose) in mse_pose_list:
            if mse < self.mse_threhsold - self.mse_threhsold/2: 
                #we are stricter with memory assumption than with pure extrapolation 
                selected_poses.append(pose[0])
                selected_poses_mse.append(mse)
            else:
                break
        
        return selected_poses, selected_poses_mse
    
    def predicted_mse_of_poses(self, place:torch.Tensor, observations:dict, pose_observations:dict)-> list:
        """ for each pose in the pose_ob' dict and the ob' place,
        we evaluate the predicted image 
        returns a list of list containing mse and pose"""
        mse_pose_list = []
        try:
            allocentric_model = self.allocentric_model.fork()
        except AttributeError:
            allocentric_model = self.allocentric_model.model

        for pose_idx in range(pose_observations['pose'].shape[1]):
            allocentric_model.reset()
            pose = pose_observations['pose'][:,pose_idx,...].unsqueeze(1)
            observations.update(pose= pose)
            mse_pose, image_predicted = mse_observation(allocentric_model, place, observations)
            mse_pose_list.append([float(mse_pose.numpy()), pose.squeeze(1).type(torch.int64).tolist()])
        return mse_pose_list
    
    def best_matching_poses_with_ob(self, place:torch.Tensor, relevant_ob:dict, num_return_poses:int=4) -> list:
        """ 
        return matching poses and their mse with the observation.
        Limited to 'num_return_poses' set to 4.
        """
        
        if 'pose' in relevant_ob:
            #IN CASE WE DON'T TRUST THE PLACE POSE ESTIMATION
            #re-estimate the goal position in this place
            x = list(range(int(relevant_ob['pose'][:,0])-1, int(relevant_ob['pose'][:,0])+2))
            y = list(range(int(relevant_ob['pose'][:,1])-1,int(relevant_ob['pose'][:,1])+2))
            theta = [int(relevant_ob['pose'][:,2])]
            pose_options = list(product(*[x,y,theta]))
            pose_options = list(map(list, pose_options)) #36 poses in total
        else:
            pose_options = self.get_place_range()
        
        best_pose_options_wt_mse = []
        for p in pose_options:
            pose_observations = self.allocentric_model.torch_observations({'pose': p}).squeeze(0)
            pose_observations.update(image = relevant_ob['image'].unsqueeze(1)) 
          
            #--- MSE EVALUATION ---#
            pose_observations = self.allocentric_model.sample_observations(pose_observations, sample= place.shape[0])
            #print('show me shape of pose and placein best matching poses with ob', pose_observations['pose'].shape, place.shape)
            model_error, image_predicted = mse_observation(self.allocentric_model.model, place, pose_observations)
            best_pose_options_wt_mse.append([float(model_error.cpu().detach().numpy()), p])

        best_pose_options_wt_mse = np.array(sorted(best_pose_options_wt_mse, key=itemgetter(0))[:num_return_poses], dtype=object)
        return best_pose_options_wt_mse

    

class AllocentricModel(PerceptionModel):
    def __init__(self, config: dict) -> None:
        PerceptionModel.__init__(self)
        model_config = config['model']
    
        model_type = model_config.get("type", None)
        if model_type == "GQN":
            self.model = GQNModel(**model_config)
            
        epoch = config.get('model_epoch', None)
        if os.path.isdir(config['params']):
            model_param = get_model_parameters(config['params'], epoch)
        else: #we consider the file as a .pt without checking
            model_param = config['params']
        
        self.model.load(model_param,map_location=config['device'])
        self.model.to(config['device'])
    
        self.space_post = None
        self.live_training = False
        self.observations_keys = config['dataset']['keys'] 
        self.observations_shape_length = {'image':5, 'action':2, 'pose':3}
     
    def get_observations_keys(self) -> list :
        return self.observations_keys
    
    def torch_observations(self,observations:dict):
        observations = torch_observations(observations, self.observations_keys)
        if 'image' in observations:       
                observations.update(image = self.resize_image(observations['image']))
        return observations

    def sample_observations(self, observations:dict, sample:int)-> dict:
        ''' all the observations are sampled to have the desired batch size'''
        observations_keys = self.observations_keys + ['action']
        obs = sample_observations(observations, observations_keys, sample, self.observations_shape_length)
        for key in obs:
            self.move_to_model_device(obs[key])   
        return obs
    
    def move_to_model_device(self, tensor):
        return tensor.to(self.model.device) 
        
    def digest(self, observations:dict, space_posterior:torch, sample:int =1, reconstruct=False): 
        #TODO: adapt for several sensors
        # returns dict with current prior/posterior
        #print('observations', observations['pose'].shape, observations['image'].shape)
        obs = self.sample_observations(observations, sample = sample)
        #NOTE: IF TRAINING continues while moving
        if not self.live_training:
            with torch.no_grad():
                step = self.model.forward(obs, place = space_posterior, reconstruct=reconstruct)
        else:
            print('uncomment process in digest pytorchslammodel')
            #step = self.model.forward(obs, place = space_posterior, reconstruct=False)
        return step

    def create_query(self, obs, keys=[]):
        observations = tensor_dict({})
        for k in keys:
            ob = obs.get(k,None)
            observations[k+'_query'] = ob
        return observations


    def predict(self, place, next_pose, sample=1):
        '''
        predict image given new pose and place
        predict new place given the expected image of that pose
        '''
        # returns dict with imagined observations for given query pose
        ob_shapes = {'pose_query':3, 'image_query':5}
        observations_keys = ['pose_query', 'image_query']

        next_pose = sample_observations(next_pose, observations_keys, sample, ob_shapes)
       
        with torch.no_grad():
            #predict pose or image given ob and current belief
            future_step = self.model.forward(next_pose, place, reconstruct=True)

            if 'image_predicted' in future_step:
                predicted_ob = 'image'
                expected_ob = 'pose'

            if 'pose_predicted' in future_step:
                predicted_ob = 'pose'
                expected_ob = 'image'

            future_step[predicted_ob] = future_step[predicted_ob+'_predicted']
            future_step[expected_ob] = future_step.get(expected_ob+'_predicted', \
                                        future_step.get(expected_ob+'_query', future_step.get(expected_ob)))
            
            #predict believe update given predicted ob 
            future_step = self.model.forward(future_step, place, reconstruct=False)
    
        return future_step


    def reset(self, state:torch.Tensor=None, post:torch.Tensor=None)-> None:
        if state is not None:
            state = state.to(self.model.device)
            post = post.to(self.model.device)
        self.model.reset(state,post)

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

    def resize_image(self, image:torch.Tensor)-> torch.Tensor:
        ''' Transforn image into model desired shape'''
        model_width = self.model.observation_size("image")[-1]
        model_height = self.model.observation_size("image")[-2]
        if model_width != image.shape[-1] or model_height != image.shape[-2]:
            image = interpolate(image, size=(model_height, model_width), mode="nearest")
    
        return image
    
