import copy

import numpy as np
import torch
import cv2


from env_specifics.env_calls import (call_env_entry_poses_assessment,
                                     call_env_number_of_entry_points,
                                     call_env_remove_double_poses,
                                     call_get_place_behind_limit)
from navigation_model.Services.allocentric_model import \
    init_allocentric_process
from dommel_library.datastructs import cat as cat_dict
from navigation_model.Services.egocentric_model import init_egocentric_process
from navigation_model.Services.memory_service.memory_graph import MemoryGraph
from navigation_model.Services.model_modules import torch_observations, sample_ob
from navigation_model.visualisation_tools import convert_tensor_to_matplolib_list, visualise_image

class Manager():
    def __init__(self, allo_model_config:dict, memory_graph_config:dict, env_actions:list, env:str, lookahead:int=5, replay_buffer=None) :

        self.empty_memory = True
        self.num_samples = 5
        self.env_relevant_ob = {}

        self.env_specific = env

        #DICT CONFIG
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        #ego config //STATIC//
        self.egocentric_process = init_egocentric_process(env_actions, device)

        #ALLO config 
        self.allocentric_process, self.std_place_th = init_allocentric_process(allo_model_config, self.env_specific, device)
        self.replay_buffer = replay_buffer
        #MemoryGraph initialisation
        self.memory_graph = MemoryGraph(**memory_graph_config) 
        self.memory_graph.experience_map.replay_buffer = self.replay_buffer

        self.observations_keys = list(set(self.allocentric_process.get_observations_keys()) | \
                                set(self.egocentric_process.get_observations_keys()))
        

        #NAVIGATION PARAMETERS
        self.default_lookahead = lookahead
        self.variable_lookahead = self.default_lookahead

  

        #TODO: change all that    
        # THE GQN + OZ PROCESS
        #self.place_model = PlaceGQN(self.allocentric_process, self.egocentric_process, self.observation_keys, std_place_th = self.std_place_th)
            
        #To be sure not to remove the ghost node realised behind oneself when entering a room
        #self.dist_margin = memory_graph_config['delta_exp_threshold'] + 1


        #goal_to reach variables
        #TODO: Goal process not in manager but in goal python file.
        #Goal will be a class instead of a dict
        #SO ALL THE BELOW ELEMENTS WILL BE ELSEWHERE
        # self.goal_data = {}
        # #How much proba we want to put on unconnected links from 1 to inf 
        # # (high number means highly improbable that they are chosen, 
        # # low number means higher choice proba)
        # self.weigth_on_unconnected_exps = 10 
        # self.wt_lower_range = np.array([235,235,235])
        
        
        # #TODO: put all that in exploration
        # #curiosity driven variables
        # #self.turn_motion_perceived = False
        # self.original_lookahead = self.place_model.lookahead 
        # self.agent_exploring = True
        # self.curiosity_temp = 100

        
        # #TESTs
        # self.n_step = 0
        # self.close_exploration_plot = True
        # self.GP_data = []
        #self.last_chosen_step_policy_EFE = None

        #self.lock = multiprocess.Lock()

#================= SETUP METHODS ==============================================================================#
   
    #TODO: add this call in test setup, not in manager
    def load_memory(self, load_memory: str) -> None:
        '''
        We re-use previously memorised experiences
        '''
        self.memory_graph.load(load_memory)
        self.empty_memory = False
    
    def save_memorised_map(self,file:str) -> None:
        self.memory_graph.save(file)
    
    def set_env_relevant_ob(self, significant_ob: dict) -> None:
        ''' save the given step observatrion as relevant ob for future use, 
        of input format [batch,...]'''

        self.env_relevant_ob = significant_ob
    
    def reset_variable_lookahead_to_default(self) -> None:
        self.variable_lookahead = self.default_lookahead

#================= GET METHODS ==============================================================================#
   
    def get_observations_keys(self) -> list :
        return self.observations_keys
    
    def get_manager_sampling(self) -> int:
        ''' return manager set sample lenght'''
        return self.num_samples
    
    def get_env_relevant_ob(self) -> dict:
        return self.env_relevant_ob  
    
    def get_allocentric_model_mse_theshold(self) -> float:
        return self.allocentric_process.get_mse_threshold()
    
    def get_confidence_about_place_description(self) -> bool:
        return self.allocentric_process.confident_about_place_description()
    
    def get_current_lookahead(self) -> int:
        return self.variable_lookahead
    
    def mse_under_threshold(self, mse: float, sensitivity:float = 1) -> bool:
        ''' 
        is the mse under the allocentric mse threshold * sensitivity?
        the sensitivity is used to adapt the threshold to our need
        '''
        return mse < self.get_allocentric_model_mse_theshold() * sensitivity

    def get_best_place_hypothesis(self) -> dict:
        return self.allocentric_process.get_best_place_hypothesis()
    def get_all_places_hypothesis(self)->dict:
        return self.allocentric_process.get_all_place_hypothesis()
    
    def agent_lost(self) -> bool:
        ''' are we lost? '''
        return self.allocentric_process.place_doubt_step_count() > 6

    def get_location_limits(self, exp_id:int=None)-> list:
        return self.memory_graph.get_exp_relevant_poses(exp_id)

    def get_connected_place_info(self, pose:list)-> tuple[any, int, list]:
        ''' 
        If we imagine going to a known place, we retrieve the place,
        If we don't know what to imagine, we create an empty place
        '''
        expected_exp_id, limit_jonction_from_other_place = call_get_place_behind_limit(self.env_specific, self, pose)
        #If known place
        if expected_exp_id >= 0 :
            expected_place = self.memory_graph.get_exp_place(expected_exp_id)
            expected_place = self.allocentric_process.place_as_sampled_Normal_dist(expected_place,self.num_samples)
        else:
            expected_place = None 
            print('no expected known experience behind limit', pose)
        
        return expected_place, expected_exp_id, limit_jonction_from_other_place

    def torch_sample_place(self, place:np.ndarray) -> torch.Tensor:
        return self.allocentric_process.place_as_sampled_Normal_dist(place,self.num_samples)
    
    def get_current_exp_id(self)->int:
        return self.memory_graph.get_current_exp_id()
    
    def get_exps_organised_by_distance_from_exp(self, exp_id:int=None)->dict:
        return self.memory_graph.get_exps_organised(exp_id)
    
    def get_all_exps_in_memory(self, wt_links:bool=True):
        return self.memory_graph.get_exps(wt_links)
    
#=================  FORMAT METHODS ==========================================================================
    def save_pose_in_memory(self, pose:list)-> None:
        ''' save pose in memory graph exp memory'''
        #TODO: this is too particular for minigrid, to generalise
        relevant_poses = list(self.memory_graph.get_exp_relevant_poses())
        if pose not in relevant_poses:
            relevant_poses.append(pose)
            print('XXXXXXX relevant_poses',relevant_poses)
            self.memory_graph.memorise_poses(relevant_poses)

    def increase_lookahead(self, max:int=6)->bool:
        '''
        Increase the lookahead if it's under the max lookahead threshold.
        return wether it increased the lookahead or not
        '''
        if self.variable_lookahead < max:
            self.variable_lookahead+=1
            return True
        return False

    def sample_visual_ob(self, observation:torch.Tensor) -> torch.Tensor:
        return sample_ob(observation, self.num_samples, len(observation.shape))
    
#================= STEP OBSERVATION UPDATE ==========================================================================
    def digest(self, sensor_data:dict) -> None:
        ''' 
        for new observations,
        update the egocentric model
        update the allocentric model --> implies possible change of state
        update the memory graph --> implies possible change location
        '''

        #=====Observations treatment process======#
        torch_sensor_data = torch_observations(sensor_data, self.observations_keys)
        print(torch_sensor_data["pose"],torch_sensor_data["image"])
        #=====ALLO+EGOCENTRIC MODELS process (update model with latest motion observations)======#
        self.egocentric_process.digest(torch_sensor_data, self.num_samples) #
        self.process_place_believes(torch_sensor_data)
        
        #======Update the map======# 
        # localize on map, update map
        self.memory_update(torch_sensor_data, self.get_best_place_hypothesis())
  
        return 
    
    def process_place_believes(self, torch_sensor_data: dict):
        ''' 
        Update the allocentric model with new data.
        It implies the creation of parallel place hypothesis if
        the model predicts the observation incorrectly
        '''
        #TODO: transform the hypothesis in classes, 1 class by info
        self.allocentric_process.update_place_believes(torch_sensor_data, self.num_samples)
        #If we are lost and there are memory graph as experiences in memory
        if self.agent_lost() and not self.empty_memory: 
            #Verify if some prev place door poseS match current ob. If yes, add them to the hypothesis
            print("adding memories to hypothesis")
            self.add_memories_to_hypothesis(torch_sensor_data)
                    
        self.allocentric_process.assess_believes()
    
    def memory_update(self, observations: dict, place_descriptor: dict) -> None:
        """ update the topo map with new step update """
        #Save previous step state
        prev_exp_id = self.memory_graph.get_current_exp_id()
        self.memory_digest(observations, place_descriptor)
        current_exp_id, current_view_cell_id = self.memory_graph.get_current_exp_and_view_id()

        print('CURRENT ID and view cell ID',current_exp_id, current_view_cell_id)
        
        #If no memory formed
        if current_exp_id == -1:
            print("there was no memory")
            return
        else:
            self.empty_memory = False
        
        #NOTE: IS THIS PART REALLY USEFULL? DON'T THINK SO, TO CHECK
        # current_view_cell_place = multivariate_distribution(self.memory_graph.view_cells.templates[current_view_cell_id])  
        #If the memory recognised an experience that do NOT match the current belief
        #if slam_obs["place"] is not None and not (current_view_cell_place == slam_obs["place"]).all():
             #self.place_model.update_place(observations, current_view_cell_place)
        
        #If we change experience, we do a recap on previous exp main features before storing it in memory
        #if current_exp_id != prev_exp_id and prev_exp_id != -1:
            #self.handle_exp_change(prev_exp_id)

    def rgb56_to_template64(self,
    img,
    eps: float = 1e-6,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
        """
        56×56×3 RGB  →  64-D descriptor
            • 48 dims = 16-bin histograms of L*, a*, b*
            • 16 dims = 4×4 block-mean edge magnitudes (Sobel)
        """

        # ------------------------------------------------------------------ #
        # 1. Make sure we have HWC uint8                                     #
        # ------------------------------------------------------------------ #
        print(f"[DBG] raw in  {type(img)}, shape={getattr(img,'shape',None)}, "
            f"dtype={getattr(img,'dtype',None)}")

        if torch.is_tensor(img):
            if img.shape == (3, 56, 56):                      # CHW tensor
                img = img.permute(1, 2, 0).contiguous().cpu().numpy()
                print("[DBG] permuted CHW→HWC, now", img.shape)
            else:
                img = img.cpu().numpy()
                print("[DBG] tensor already HWC → numpy")
        else:
            if img.shape == (3, 56, 56):                      # CHW numpy
                img = np.transpose(img, (1, 2, 0))
                print("[DBG] numpy CHW→HWC, now", img.shape)

        assert img.shape == (56, 56, 3), f"expected 56×56×3, got {img.shape}"

        if img.dtype != np.uint8:
            img = (img * 255.0).round().astype(np.uint8)
            print("[DBG] scaled to uint8")

        # ------------------------------------------------------------------ #
        # 2. 16-bin Lab histograms (48 dims)                                 #
        # ------------------------------------------------------------------ #
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
        L   = (lab[:, :, 0] * 255.0 / 100.0).clip(0, 255)
        a   = lab[:, :, 1] + 128.0
        b   = lab[:, :, 2] + 128.0

        bins = np.linspace(0, 256, 17, dtype=np.float32)
        h_L, _ = np.histogram(L, bins=bins)
        h_a, _ = np.histogram(a, bins=bins)
        h_b, _ = np.histogram(b, bins=bins)

        h48 = np.concatenate([h_L, h_a, h_b]).astype(np.float32)
        h48 /= h48.sum() + eps
        print(f"[DBG] hist L1-norm={h48.sum():.3f}")

        # ------------------------------------------------------------------ #
        # 3. 4×4 Sobel-edge energy (16 dims)                                 #
        # ------------------------------------------------------------------ #
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mag  = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        edge16 = [
            mag[y : y + 14, x : x + 14].mean()
            for y in range(0, 56, 14)
            for x in range(0, 56, 14)
        ]
        edge16 = np.asarray(edge16, np.float32)
        edge16 /= edge16.sum() + eps

        # ------------------------------------------------------------------ #
        # 4. Concatenate → 64-D  & return torch tensor                       #
        # ------------------------------------------------------------------ #
        vec64 = np.concatenate([h48, edge16])
        print("[DBG] vec64  shape", vec64.shape, " first5", vec64[:5])

        return torch.from_numpy(vec64).to(device)
        

    
    def memory_digest(self, observations: dict, place_descriptor: dict) -> None:
        """ We update the memory graph with current:
        - believed place (dist)
        - believed pose ([x,y,th])
        - action ([F, R, L])
        with current believed place + pose not NONE only if we are confident about the place
        We also update the belief over the place if the memory graph believes we are
        in another experience.
        """
        #=== UPDATE MAP WITH CURRENT BELIEF ===#
        slam_obs = {"place": None, 'pose': None, "HEaction": observations['action'].cpu().detach().numpy(), "image":None}
        print("This is the place descriptor [0] that updates the memory graph", place_descriptor["pose"])
        print("this is the stable distribution in std place th compared to the place descriptor std ", self.std_place_th)
        #if we have only 1 place hypothesis and we have a stable distribution
        if (not self.agent_lost() and place_descriptor['std'] < self.std_place_th):
            print("slam before changes",place_descriptor['post'].shape )
            slam_obs["place"] = torch.mean(place_descriptor['post'],dim=0).squeeze(0)
            print("slam obs place", slam_obs["place"].shape)
            slam_obs["pose"] = place_descriptor['pose']
            print("slam obs image shape", observations["image"],observations["image"].shape)
            slam_obs["image"] = self.rgb56_to_template64(
                                observations["image"]) 
            print("slam obs image",slam_obs["image"], slam_obs["image"].shape)
            
        else :
            self.memory_graph.memorise_poses([]) #if we are lost, we reset door memories
            
        #process state in memory_graph model
        print("maybe slam",slam_obs)
        self.memory_graph.digest(slam_obs, dt=1, adjust_map = False)

    def handle_exp_change(self, prev_exp_id:int)-> None:
        """ we change experience, thus we save the important info relative to this place"""

        #prev_place_id = self.memory_graph.get_exp_view_cell_id(prev_exp_id)
        print('update previous place limit poses')
        self.update_place_limits(prev_exp_id)
        
        if self.memory_graph.ghost_node_process :
            #We create ghost nodes at memorised poses + margin: memory_graph_config['delta_exp_threshold']+ 1 
            self.memory_graph.create_ghost_exps(exp_id = prev_exp_id) 

#================= Memories Methods ==========================================================================#
    def update_place_limits(self, exp_id:int=-1):
        '''
        Given exp id, get the place and find if any relevant obs found there
        If yes, memorise them in long term memory (not essential, but easier on process)
        '''
        prev_place = self.memory_graph.get_exp_place(exp_id)
        #We serach for poses matching the desired observation
        ob_poses = self.identify_observation_in_place(prev_place, self.env_relevant_ob)
        
        #We save the previous experience door poses in memory
        
        self.memory_graph.memorise_poses(ob_poses, exp_id)
        #do we create ghost experiences behind identified doors?

   
    def ascii_hist(self,data, bins=20, width=40):
        """
        Create an ASCII histogram for a 1D numpy array.
        """
        
        hist, bin_edges = np.histogram(data, bins=bins)
        max_val = hist.max()
        for i in range(len(hist)):
            edge = bin_edges[i]
            count = hist[i]
            # Normalize the count to the desired width
            bar = '#' * int((count / max_val) * width) if max_val > 0 else ''
            print(f"{edge:8.2f} | {bar} ({count})")
    
    def ascii_histogram_adaptive(self,samples, bins=20):
        """
        Adaptively create ASCII histograms based on the shape of the samples.
        If samples is 1D, prints one histogram.
        If samples is 2D (num_samples x d), prints one histogram per column.
        If samples has more dimensions, it flattens the data.
        """
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        
        # Remove singleton dimensions (e.g., if shape is (num_samples, 1, d))
        samples = np.squeeze(samples)
        
        if samples.ndim == 1:
            print("Histogram for all values:")
            self.ascii_hist(samples, bins=bins)
        elif samples.ndim == 2:
            num_dims = samples.shape[1]
            for i in range(num_dims):
                print(f"Histogram for dimension {i}:")
                self.ascii_hist(samples[:, i], bins=bins)
                print()  # Blank line for separation
        else:
            print("Data has more than 2 dimensions; flattening:")
            self.ascii_hist(samples.flatten(), bins=bins)


    #TODO: change how we check for exp plausibility for competition
    def add_memories_to_hypothesis(self, observations):
        '''
        we search among the experiences of the memory graph if any door entry fits current observation.
        if yes then we add this observation in the competition of the fitest hypothesis to describe the place
        '''
        #WE want the exps organised by current GP
        exps = self.memory_graph.get_exps_organised(from_current_pose=True)
        #we go through all memorised experiences
        for exp in exps:
            if exp['id'] < 0:
                continue
            print('exp', exp['id'], ' distance to current GP', exp['delta_goal'], 'MAX TH:', self.memory_graph.get_delta_exp_threshold() * 2)
            #If the exp is too far from last
            if exp['delta_goal'] > self.memory_graph.get_delta_exp_threshold() * 2: #NOTE:THIS NUMBER IS PURELY ARBITRARY
                print("we got breaken")
                break
            #we recall the experience entry positions
            #TODO:change this logic of door poses in memory graph exp as such
            entry_poses = copy.deepcopy(exp['observation_entry_poses'])
            print("ENTRY_POSES", entry_poses)
            if len(entry_poses) == 0:
                print("we continued")
                continue
            entry_poses = call_env_entry_poses_assessment(self.env_specific, entry_poses)
            place = self.allocentric_process.place_as_sampled_Normal_dist(exp['observation'], self.num_samples)
            #print("Visualizing the sampled place distribution:")
            #self.ascii_histogram_adaptive(place, bins=10)
            observations = self.allocentric_process.sample_observations(observations, self.num_samples)
            #print("observations", observations)
            pose_observations = self.allocentric_process.create_pose_observations(entry_poses, self.num_samples)
            #print("this pose_observations",pose_observations)
            plausible_poses, poses_mse = self.allocentric_process.assess_poses_plausibility_in_place(place, observations, pose_observations)
            
            print('exp', exp['id'], 'selected_poses', plausible_poses, 'associated mse:', poses_mse)
            if len(plausible_poses)>0:
                self.allocentric_process.add_hypothesis_to_competition(place, poses = plausible_poses, exp= exp['id'], mse = poses_mse)

    def identify_observation_in_place(self, place:np.ndarray, env_relevant_ob:dict) -> list:
        ''' search for a particular observaion in place '''
        door_poses = []
        if place is None:
            place = self.get_best_place_hypothesis()['post']
        else:
            place = self.allocentric_process.place_as_sampled_Normal_dist(place,self.num_samples)
        #How many entry points do we want to remember
        num_return_poses = call_env_number_of_entry_points(self.env_specific)
        print("this should be 4", num_return_poses)
        #best poses matching ob
        best_poses_wt_mse = self.allocentric_process.best_matching_poses_with_ob(place, env_relevant_ob, num_return_poses)
        print(best_poses_wt_mse)
        #remove duplicate pose (in minigrid means removing double orientation)
        ob_poses = call_env_remove_double_poses(self.env_specific, best_poses_wt_mse)
        print(ob_poses)
        #Only remember pose if it is under a mse threshold
        for mse, p in ob_poses:
            if self.mse_under_threshold(mse, sensitivity=0.8) :
                door_poses.append(p)
        print('predicted door poses 222', door_poses)
        print('corresponding MSE ', ob_poses[0:len(door_poses)])
        return door_poses
    
    def convert_orientation_between_two_places_ref_frame(self,pose:list,  start_exp_id:int= None, goal_exp_id:int=None)-> int:
        return self.memory_graph.convert_pose_orientation_from_start_ref_frame_to_another(pose,  start_exp_id, goal_exp_id)

    def connect_places_to_current_location(self)->dict:
        ''' 
        Return all the exps linked to current exp
        '''
        linked_exps_info = {}
        current_exp_door_poses = self.memory_graph.get_exp_relevant_poses()
        #NOTE: This means no update of door poses with new seen door poses.
        #We consider that they are all imagined during exploration
        if len(current_exp_door_poses) == 0:
            self.update_place_limits(exp_id=-1)
            current_exp_door_poses = self.memory_graph.get_exp_relevant_poses()
        
        for current_exp_door_pose in current_exp_door_poses:
            linked_exp_id, door_pose_from_new_place = self.memory_graph.linked_exp_behind_door(current_exp_door_pose)

            if linked_exp_id >= 0 :
                view_cell = self.memory_graph.get_exp_view_cell(linked_exp_id)
                #the decay increase at each step in place and decrease over time as we move in other places
                view_cell_place_decay = view_cell.decay
                linked_exps_info[len(linked_exps_info)] = {
                                    'linked_exp_id' : linked_exp_id, \
                                    'current_exp_door_pose':current_exp_door_pose, \
                                    'door_pose_from_new_place': door_pose_from_new_place,\
                                    'view_cell_place_decay':view_cell_place_decay}
                #linked_exps_info.append([current_exp_door_pose, linked_exp_id, door_pose_from_new_place, view_cell_place_decay])
        print('linked_exps_info', linked_exps_info)

        return linked_exps_info
    
    def connected_place_to_visit(self,)-> dict:
        ''' 
        Return the most decayed place connected to current exp, if None, it returns an empty dict
        '''
        linked_exps_info = self.connect_places_to_current_location()
        if linked_exps_info:
            place_to_go = min(linked_exps_info.values(), key=lambda x: x['view_cell_place_decay'])
            
        else:
            place_to_go = {}
            print('No connected exp to current exp')
    
        print('place_to_go (most decayed place)', place_to_go)
        
        return place_to_go

    def get_egocentric_posterior(self)-> torch.Tensor:
        return self.egocentric_process.get_egocentric_posterior()
#================= Predictions Methods =================================================================#
    def single_pose_allocentric_prediction(self, pose:list, place) -> dict:
        """
        given a place and pose, what does the allo-model predicts for that step
        """

        return self.allocentric_process.allocentric_pose_prediction(pose, place , self.num_samples)
    
    def several_poses_allocentric_prediction(self, poses:list, place) -> dict:
        """
        given a place and list of pose (as list), what does the allo-model predicts for that step
        This is so the place updating resulting from the step prediction is unique for each pose of the list
        """
        results = None
        for p in poses:
            step_pred = self.single_pose_allocentric_prediction(p, place)

            if results is None:
                results = step_pred
            else:
                results = cat_dict(results, step_pred)
        return results
    
    def policy_egocentric_prediction(self, policy:list) -> tuple[list, dict]:
        if len(policy) == 0:
            return [], {}
        policy = torch.as_tensor(policy)
        policy, ego_prediction = self.egocentric_process.egocentric_policy_assessment(policy, self.num_samples, prediction_wanted=True)
        return policy.tolist(), ego_prediction
#================ Policies ====================================================================#

    def get_plausible_policies(self,policies: list) -> list:
        """  
        return only one set of all the policies 
        that seem dynamically plausible to effectuate to the egocentric model
        """
        plausible_policies_dict = {}
        for policy in policies:
            policy = torch.as_tensor(policy)
            # ==== egocentric model PROCESS ==== #
            #assess policy and cut off any colision move
            policy = self.egocentric_process.egocentric_policy_assessment(policy, self.num_samples)
            #if no action to apply, we don't need to consider it
            if policy.shape[0] != 0:
                #We want only one set of each policy, thus avoiding any double occurence
                #if we have only 1 action (list)
                if len(policy.shape) == 1:
                    policy_tuple = tuple(policy.tolist())
                #if we have a sery of actions (list of list)
                else:
                    policy_tuple = tuple(map(tuple, policy.tolist()))
                plausible_policies_dict[policy_tuple] = policy
        
        plausible_policies = list(plausible_policies_dict.values())
        return plausible_policies


#====== Data TEST related functions ======#

    def exp_visualisation_proba(self, place_descriptors):
        '''
        This is for testing purposes
        
        '''   
        hypo_weights = {}
        new_exp = -1
   
        for id, content in place_descriptors.items():
            
            if 'exp' in content :
                exp = content['exp']
                new_exp = exp
            else:
                new_exp-=1
            if new_exp in hypo_weights:
                if hypo_weights[new_exp] < content['hypothesis_weight']:
                    hypo_weights[new_exp] = content['hypothesis_weight']
            else:
                hypo_weights[new_exp] = content['hypothesis_weight']
        
        
        ids = list(hypo_weights.keys())
        weights = np.array(list(hypo_weights.values())) + 0.01 #to avoid 0

        
        # prob = Categorical(logits= torch.from_numpy(weights)).probs.numpy()
        # if len(place_descriptors) >1:
        #     fig, ax = plt.subplots(1 ,figsize=(9, 3), layout='constrained')
        #     plt.title(f"Hypothesis weight")
        #     ax.bar(ids, prob)
        #     plt.show()


        #new_exps_prob = []

        # for i in range(len(ids)):
        #     if ids[i] < 0:
        #         new_exps_prob.append([prob[i], ids[i]])
        # print(prob)
        # print(new_exps_prob)
        # print('mean prob of all new exps', np.mean(new_exps_prob, axis=0)[0])
        # max_prob = np.amax(new_exps_prob, axis=0)[0]
        # ids_index = np.where(np.array(new_exps_prob)[:,0] == max_prob)[0][0]

        # real_max_prob = np.max(prob)
        # print('max exp prob', real_max_prob, ids.index[real_max_prob] )
      
        # max_ids = ids[ids_index]
        # print('MAX prob of new exps', max_prob,max_ids)
        
        return ids, list(weights)
          
    def get_setting_variables(self):
        ''' for visualisation purposes'''
        dict_var = {'lookahead': self.place_model.lookahead, 'x_range': self.place_model.x_range, 'y_range': self.place_model.y_range, \
            'mse_err_th': self.place_model.mse_err_th , 'temp_GQN': self.curiosity_temp, 'weigth_exp': self.weigth_on_unconnected_exps, \
            'match_th': self.memory_graph.view_cells.MATCH_THRESHOLD, 'delta_exp': self.memory_graph.experience_map.DELTA_EXP_THRESHOLD, \
            'ghost_node': self.memory_graph.ghost_node_process, 'wt_lower_range': self.wt_lower_range}
        
        return dict_var

    def set_setting_variables(self, dict_var):
        ''' to modify the variables during a test '''
        self.place_model.lookahead = dict_var.get('lookahead', self.place_model.lookahead)
        x = dict_var.get('x_range', self.place_model.x_range)
        y = dict_var.get('y_range', self.place_model.y_range)
        self.place_model.pose_options = self.place_model.pose_option_setup(x,y)
        self.place_model.x_range, self.place_model.y_range = x,y
        self.place_model.mse_err_th = dict_var.get('mse_err_th', self.place_model.mse_err_th)
        self.curiosity_temp = dict_var.get('temp_GQN', self.curiosity_temp)
        self.memory_graph.view_cells.MATCH_THRESHOLD = dict_var.get('match_th', self.memory_graph.view_cells.MATCH_THRESHOLD )
        self.memory_graph.experience_map.DELTA_EXP_THRESHOLD = dict_var.get('delta_exp',self.memory_graph.experience_map.DELTA_EXP_THRESHOLD)
        self.memory_graph.ghost_node_process = dict_var.get('ghost_node',self.memory_graph.ghost_node_process)
        self.weigth_on_unconnected_exps = dict_var.get('weigth_exp',self.weigth_on_unconnected_exps)
        self.wt_lower_range =  np.array(dict_var.get('wt_lower_range', self.wt_lower_range ))
    
        #plt.show()

    # def modify_place_model_variables(self,**kwargs):
    #     for key, value in kwargs.items():
    #         setattr(self.place_model, key, value)

    #     if len(set(kwargs.keys()).intersection(('x_range', 'y_range')))>0:
    #         pose_opt = self.place_model.pose_option_setup(self.place_model.x_range, self.place_model.y_range)
    #         self.place_model.pose_options = pose_opt

    #NOTE: IS THIS METHOD USEFULL?
    def get_nodes_info(self):
        ''' return the number of exp created + how many links each have.
        Node and not exp as its from the env perspective'''
        exps_list = self.memory_graph.get_exps(wt_links = True)

        n_nodes = len(exps_list)
        links_per_node = []
        for exp in exps_list:
            links_per_node.append(len(exp['links']))

        return n_nodes, links_per_node
    

    def given_pose_prediction_visualisation(self, pose):
        """ visuliasing tool to see what the model predicts for a particular pose
        """
        place_info = self.get_best_place_hypothesis()
        step = self.allocentric_process.allocentric_pose_prediction(pose = pose, place = place_info['post'], num_samples=self.num_samples)
        
        visualise_image(torch.mean(step['image_predicted'], dim=0), 'pose:'+ str(pose), fig_id = 100, )

        return step['image_predicted']
        