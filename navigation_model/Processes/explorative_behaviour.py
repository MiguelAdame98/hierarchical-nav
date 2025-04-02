from itertools import product
import numpy as np
import torch
from torch.distributions.categorical import Categorical

from env_specifics.env_calls import call_process_limit_info
from env_specifics.minigrid_maze_wt_aisles_doors.minigrid_maze_modules import (
    from_door_view_to_door_pose, is_agent_at_door_given_ob)
from navigation_model.Processes.AIF_modules import (calculate_KL,
                                                    estimate_a_surprise)
from navigation_model.Processes.motion_path_modules import (
    action_to_pose, create_policies, define_policies_objectives)
from navigation_model.visualisation_tools import (
    transform_policy_from_hot_encoded_to_str, visualise_image)


class ExplorativeBehaviour():
    def __init__(self, possible_actions: list, verbose:bool=True) :
        self.possible_actions = possible_actions
        self.verbose = verbose
        self.place_info_gain = []
        self.latest_EFE = None

        self.agent_exploring = True
        self.rolling_info_gain = []

    def set_latest_policy_EFE(self, G:float)-> None:
        self.latest_EFE = G

    
#====================== Get Methods ===================================================================
    def define_is_agent_exploring(self, info_gain_coeff:float, policy_G:float, threshold:int=1)-> bool:
        """ 
        according to the info gain coeff, the policy_G and the threshold
        return whether the agent is stuck in a local minima 
        and doesn't explore anymore
        The agent explore if it is surprised enough in env or it has an idea where to explore next
        For teh agent to reach an info gain coeff < 0.001, it is expected to need about 10-30steps in a room
        """
        print("why no explore?",abs(info_gain_coeff),(threshold / 500) ,abs(info_gain_coeff) < (threshold / 500),policy_G < (threshold/ 5),policy_G ,(threshold/ 5)  )
        #if (abs(info_gain_coeff) < (threshold / 1000) and policy_G < (threshold/ 10)) :
        if (abs(info_gain_coeff) < (threshold / 400) and policy_G < (threshold/ 5)) :
            self.agent_exploring = False
        else:
            self.agent_exploring = True

        if self.verbose:
            print('is agent exploring', self.agent_exploring,' updated with info gain coeff', info_gain_coeff, \
                  'vs TH', threshold / 1000, 'policy', policy_G, 'vs TH', threshold/ 10)
        return self.agent_exploring
     
    def is_agent_exploring(self)-> bool:
        """ 
        return whether the agent is stuck in a local minima 
        and doesn't explore anymore
        """
        return self.agent_exploring 
         
    def get_latest_policy_EFE(self, reset_EFE:bool = False)->float:
        """
        Retrieve latest generated EFE, 
        if reset_EFE is True then we erase it upon return
        """
        EFE = self.latest_EFE
        if reset_EFE:
            self.latest_EFE = None
        return EFE
    
    def get_policies_info_gain(self, manager:object, policies:list)->tuple[list,list]:
        """ 
        info gain based motion.
        the lowest G based on KL gets it all, the OZ model cut action 
        sequences based on its collision detection, then each action 
        is passed through the allo model pred
        """
        #possible policies are returned as a list of Tensor
        policies = manager.get_plausible_policies(policies)
        paths_KL = []
        imagined_doors_info = self.initialise_place_doors_knowledge(manager)
        for policy in policies:
            path_steps_KL,imagined_doors_info = self.path_predicted_info_gain(manager,policy,imagined_doors_info)
            paths_KL.append(np.mean(path_steps_KL))
        return policies, paths_KL
    
    def moving_average_info_gain(self, window:int=5)->float:
        """
        Calculates the average of the info_gain over a fixed window of time.
        Not ideal with info_gain, would be more adapted to KL measuring instead
        """
        window = min(window, len(self.rolling_info_gain))
        if len(self.rolling_info_gain) < window:
            ma_info_gain = 1
        else:
            ma_info_gain = sum(self.rolling_info_gain) / window
        return ma_info_gain
    
    def rolling_coeff_info_gain(self, window:int=5)->float:
        """
        Calculates the coefficient over a rolling window of info_gain.
        only if the rolling_info_gain list is at least as big as the window.
        else returns a static high value s= 10 
        """
        window = min(window, len(self.rolling_info_gain))
        if len(self.rolling_info_gain) < window:
            rc_info_gain = 10
        else:
            rc_info_gain = (self.rolling_info_gain[-1] - self.rolling_info_gain[0]) / window
        return rc_info_gain
    def update_rolling_info_gain(self, info_gain:float, window:int = 10)->None: 
        ''' 
        we store the info gain in a memory list of size window.
        The info_gain is stored up to windows size (with newest as last and oldest as first)
        when window size reached, the 1st (oldest) value pop up 
        '''
        self.rolling_info_gain.append(info_gain)
        #we only want a sliding X steps info gain
        while len(self.rolling_info_gain) > window and len(self.rolling_info_gain)>0:
            del self.rolling_info_gain[0] 

        if self.verbose:
            print('updated rolling info gain', self.rolling_info_gain)

#====================== Create Policies =============================================================
    
    def create_policies(self, manager:object, lookahead:int)-> list:
        full_exploration = manager.get_confidence_about_place_description()
        current_pose = manager.get_best_place_hypothesis()['pose'].copy()
        print('In exploration create_policies, verifying if confident about place description', full_exploration)
        goals = define_policies_objectives(current_pose, lookahead, full_exploration)
        policies = create_policies(current_pose, goals)
        return policies

#====================== Exploration METHODS===============================================================

    def one_step_ego_allo_exploration(self,manager:object):
        """
        create policies, assess their info gain and select the policy 
        that would modify the believes the most if proven true by ob (worth checking)
        return the policy and how many consecutive actions we want to apply from it
        Since it's one step exploration. We only apply the first action of the policy
        """
        lookahead = manager.get_current_lookahead()
        policies = self.create_policies(manager, lookahead)
        policies, paths_KL = self.get_policies_info_gain(manager,policies)
        chosen_policy, policy_G = self.curiosity_driven_policy_selection(policies, paths_KL)
        self.set_latest_policy_EFE(policy_G)

        return chosen_policy, 1
    
    def curiosity_driven_policy_selection(self, policies:list, paths_KL:list) -> list:
        """
        select lowest KL policy amonst all the choices.

        """
        Gs = paths_KL
        #TODO: curiosity temp do not exist
        categorised_G = Categorical(logits= self.curiosity_temp * torch.tensor(Gs) )
        index = categorised_G.sample()
        policy_G =  paths_KL[index]
        policy = policies[index]
        if self.verbose :
            print('IN Exploration: curiosity_driven_policy_selection')
            print('stochastic curiosity policy, selected G:', policy_G , ',policy:', transform_policy_from_hot_encoded_to_str(policy.tolist()))
            print('The highest G was:',max(paths_KL))
            print('______________')
        return policy, policy_G
    #based on ego_step_decision
    def one_step_egocentric_exploration(self, manager:object, lookahead:int=5):
        """
        Use the egocentric model with a X steps lookahead to move the agent
        The policies considered not leading in a wall are considered for a G assessment
        then G is Cat and softmax and we consider a third of the selected policy to move
        the agent out of current pose
        """
        policies = self.create_policies(manager, lookahead)
        ego_posterior = manager.get_egocentric_posterior()
        Gs = []
        considered_policies = []
        print('IN one_step_egocentric_exploration')
        for policy in policies:
            policy, ego_prediction = manager.policy_egocentric_prediction(policy)
            if len(policy) > 0:
                ego_posterior_sampled = ego_posterior.unsqueeze(1).repeat(1,len(policy),1)
                kl, ambiguity, _ = estimate_a_surprise(ego_prediction['prior'], preferred_state= ego_posterior_sampled,
                                    predicted_obs=ego_prediction['image_reconstructed'], 
                                    preferred_ob=torch.zeros_like(ego_prediction['image_reconstructed']))
                
                if np.isinf(ambiguity):
                    ambiguity = -2*len(policy)
                #Kl is postive, ambiguity is negative
                Gs.append(-(kl + ambiguity))
                print(len(Gs)-1, Gs[-1],transform_policy_from_hot_encoded_to_str(policy))
                considered_policies.append(policy)

        categorised_G = Categorical(logits= -0.7 * torch.tensor(Gs)) 
        index_best_option = categorised_G.sample()
        print('Chosen policy and Gs', Gs[index_best_option], transform_policy_from_hot_encoded_to_str(considered_policies[index_best_option]))
        chosen_policy = considered_policies[index_best_option]
        num_action_to_apply = int(np.ceil(len(chosen_policy)/3))
        print('applying ', num_action_to_apply, ' 1st actions from this policy')
        #We didn't average the G per step before to favorise long policies. 
        self.set_latest_policy_EFE(Gs[index_best_option]/len(chosen_policy))
        return chosen_policy, num_action_to_apply
    
#====================== RESOLVING HESITATION OVER PLACE ==================================================
    def solve_doubt_over_place(self, manager:object) -> list:
        """ 
        Consider allo and ego models to converge to
        a single place description         
        """
        
        plausible_actions, ego_predicted_images = self.define_plausible_actions(manager)
   
        hypothesis_predictions, best_hypothesis = self.solve_hypothesis(manager, plausible_actions,ego_predicted_images)
        
        if len(best_hypothesis) == 0:  
            print('no hypothesis standing from the rest, considering them all')  
            best_hypothesis = list(hypothesis_predictions.keys())
        print('Best hypotheses', best_hypothesis)

        kl_per_action, summed_ambiguity_per_action, actions_weight = \
            self.consider_kl_ambiguity_per_action(hypothesis_predictions, best_hypothesis, len(plausible_actions))
        log_kl_per_action = self.log_kl(kl_per_action)
        G_policies = self.calculate_policies_EFE(actions_weight, log_kl_per_action, summed_ambiguity_per_action)

        best_action = np.argmin(G_policies)
        chosen_action = plausible_actions[best_action]
        if self.verbose:
            print('_')
            print('solving doubt over place')
            print(len(best_hypothesis),'hypothesis being considered',len(plausible_actions))
            print('kl_per_action:', kl_per_action)
            print('summed_ambiguity_per_action:', summed_ambiguity_per_action)
            print('actions_weight', actions_weight)
            print('For policies', plausible_actions, ', EFE:', G_policies)
            print('Chosen action', chosen_action)
            print('_')

        return [chosen_action], 1
    
    def define_plausible_actions(self, manager:object) -> tuple[list, list]:
        '''
        determine which actions are doable and what is the ego prediction for those
        '''
        plausible_actions = []
        ego_predicted_images = []
        for action in self.possible_actions:
            action, ego_prediction = manager.policy_egocentric_prediction([action])
            if any(isinstance(el, list) for el in action):
                action = action[0] #just remove extra list
            if len(action) >0:
                plausible_actions.append(action)
                ego_predicted_images.append(ego_prediction['image_reconstructed'].squeeze(1))
        return plausible_actions, ego_predicted_images
    
    def solve_hypothesis(self, manager:object, plausible_actions:list, ego_predicted_images:list) -> tuple[dict, list]:
        """  
        We want to know the prediction of each hypothesis place and pose for each policy
        If we estimate the prediction to be good:
        what is the expected change to the place's hypothesis? 
        How sure is the hypothesis about its output?
        it is stored ina  dict.

        We also return a list of the best hypothesis solving all policies.
        mean Time: 9.55s / 254hypothesis
        """
        
        best_hypothesis = []
        hypothesis_predictions = {}
        place_hypothesis = manager.get_all_places_hypothesis()
        print('How many hypothesis to process:', len(place_hypothesis))
        
        for place_description in place_hypothesis.values():
            hypo_idx = len(hypothesis_predictions)

            for i, action in enumerate(plausible_actions):
                starting_pose = place_description['pose'].copy()
                pose_queries = action_to_pose(action, starting_pose)
                hypo_prediction = manager.single_pose_allocentric_prediction(pose_queries, place_description['post'][:])
                kl, ambiguity, pred_expected_error = estimate_a_surprise(place_description['post'][:], hypo_prediction['place'], hypo_prediction['image_predicted'][:,0], ego_predicted_images[i])
                

                if manager.mse_under_threshold(pred_expected_error, sensitivity=1):
                    if not hypo_idx in hypothesis_predictions:
                        hypothesis_predictions[hypo_idx] = {}
                    if kl > 10:
                        kl = 10
                    if np.isinf(ambiguity):
                        ambiguity = -2
                    hypothesis_predictions[hypo_idx][i] = {'kl': kl, 'ambiguity': ambiguity}

            #if this hypothesis predicts all actions well
            if hypo_idx in hypothesis_predictions and len(hypothesis_predictions[hypo_idx].keys()) == len(plausible_actions):
                best_hypothesis.append(hypo_idx) 

        return hypothesis_predictions, best_hypothesis

    def solve_hypothesis_option2(self, manager:object, plausible_actions:list, ego_predicted_images:list) -> tuple[dict, list]:
        """  
        OPTION 2 OF THE CODE, REALISED TO SPEED COMPUTING. mean Time: 9.26s / 254hypothesis
        ONLY 0.3s faster but less readable. So not used (kept as ref)
        We want to know the prediction of each hypothesis place and pose for each policy
        If we estimate the prediction to be good:
        what is the expected change to the place's hypothesis? 
        How sure is the hypothesis about its output?
        it is stored ina  dict.

        We also return a list of the best hypothesis solving all policies.
        """
        best_hypothesis = []
        hypothesis_predictions = {}
        place_hypothesis = manager.get_all_places_hypothesis()
        print('How many hypothesis to process:', len(place_hypothesis))
        for place_description in place_hypothesis.values():
            starting_pose = place_description['pose'].copy()
            pose_queries = [list(action_to_pose(a, starting_pose)) for a in plausible_actions]
            hypo_prediction = manager.several_poses_allocentric_prediction(pose_queries, place_description['post'][:])
            kl_amb_pred_errors = [estimate_a_surprise(place_description['post'][:], hypo_prediction['place'][:,action_id], hypo_prediction['image_predicted'][:,action_id], ego_predicted_images[action_id]) \
                                for action_id in range(len(plausible_actions))]
            #get the index of the actions giving predictions under the mse threshold
            valid_actions = [i for i, (_, _, pred_err) in enumerate(kl_amb_pred_errors) \
                            if manager.mse_under_threshold(pred_err, sensitivity=1)]
            #if any actions are valid, we consider adding them to the dict
            if len(valid_actions) > 0:
                hypo_idx = len(hypothesis_predictions)
                hypothesis_predictions[hypo_idx] = {
                    id_action: {'kl': kl, 'ambiguity': -2 if np.isinf(ambiguity) else ambiguity}
                        for id_action, (kl, ambiguity, _) in zip(valid_actions, kl_amb_pred_errors)
                    }

            #if this hypothesis predicts all actions well
            if len(valid_actions) == len(plausible_actions):
                best_hypothesis.append(hypo_idx)
        return hypothesis_predictions, best_hypothesis
    
    def consider_kl_ambiguity_per_action(self,hypothesis_predictions:dict, best_hypothesis:list, len_plausible_actions:int) -> tuple[list,list,list]:
        """
        determine the Kl and ambiguity of each action of the considered hypothesis
        We want to know the weight of each action (how many time it resulted in 
        an estimated correct prediction) 
        """
        
        actions_in_each_hypo = [list(ids.keys()) for ids in hypothesis_predictions.values()] 
        kl_per_action = [[] for _ in range(len_plausible_actions)]
        summed_ambiguity_per_action = [0]*len_plausible_actions
        actions_weight = [0]*len_plausible_actions
        for idx_hypo in best_hypothesis:
            for id_action in actions_in_each_hypo[idx_hypo]:
                actions_weight[id_action]+=1
                kl_per_action[id_action].append(hypothesis_predictions[idx_hypo][id_action]['kl'])
                summed_ambiguity_per_action[id_action] += hypothesis_predictions[idx_hypo][id_action]['ambiguity']

        return kl_per_action, summed_ambiguity_per_action, actions_weight

    def log_kl(self, kl_per_action:list) -> list:
        ''' 
        in order to homogeneise the KL and to have comparable values we log them 
        If any value is nan, we convert it to 0.
        '''
        #if no hypothesis valorise an action, we create a 0 value
        #so the list is not empty or full of nan
        kl_per_action = [sublist if (sublist and not np.isnan(sublist).all()) else [0] for sublist in kl_per_action]
        min_kl = np.min(list(map(np.nanmin, kl_per_action)))
        #we want the min value to be 0, not under, 
        # thus before doing a log we make sure no value is under 1
        if min_kl < 1:
            log_kl_per_action = [[np.log(kl + 1 - min_kl) if not np.isnan(kl) else 0 for kl in sub_list] for sub_list in kl_per_action]
        #we are also converting any np.nan value to 0 if there is any
        else:
            log_kl_per_action = [[np.log(kl) if not np.isnan(kl) else 0 for kl in sub_list] for sub_list in kl_per_action]
        return log_kl_per_action

    def calculate_policies_EFE(self,actions_weight:list, log_kl_per_action:list, summed_ambiguity_per_action:list)-> list:
        """
        For each action we compute the sum of KL multiplied by the weight of the action + 
        the ambiguity reduced by the total number of action 
        Thus to armonise the two values and have a total value that consider both arguments
        """
        G_policies = []
        tot_num_actions_considered = np.sum(actions_weight)
        for i in range(len(actions_weight)):
            if actions_weight[i] == 0:
                G_policies.append(0)
                continue
            percentage_action = actions_weight[i] / tot_num_actions_considered

            G = - np.sum(log_kl_per_action[i]) * percentage_action + summed_ambiguity_per_action[i]/ tot_num_actions_considered
            G_policies.append(G)
        return G_policies

class Exploration_Minigrid(ExplorativeBehaviour):
    def __init__(self, env_type:str, possible_actions:list, curiosity_temp:float=100, verbose:bool=True,) :
        ExplorativeBehaviour.__init__(self, possible_actions, verbose)
        self.env_type = env_type
        
        self.curiosity_temp = curiosity_temp
    
        #TEST MODULE
        #set to True to visualise generated policies
        self.visualise_policy = False
        self.test_id = 11

    def setting_prediction_and_comparison_places(self, manager:object, \
                                                 changing_place:int, imagined_doors_info:dict)\
                                                    -> tuple[torch.Tensor, torch.Tensor] :
        '''
        Are we passing a door? 
        if so we want to use other place to predict observations : prediction_post
        is the other place new? 
        if so we want to use current place to measure the surprise of new room 
        We are not passing a door:
        well then prediction and comparison place are the same
        '''
        #If we are passing a door, we use the predicted place as posterior for prediction and info gain 
        if changing_place >= 0:
            posterior_for_kl_ref = imagined_doors_info[changing_place]['connected_place']
            if posterior_for_kl_ref is None:
                posterior_for_kl_ref = manager.get_best_place_hypothesis()['post'][:]
                normal_dist_place = np.concatenate((np.zeros(32), np.ones(32)), axis=0).astype(int)
                prediction_post = manager.torch_sample_place(normal_dist_place)  #  multivariate_distribution(torch.zeros(1, 1, 32), torch.ones(1, 1, 32))
            else:
                prediction_post = posterior_for_kl_ref
            return posterior_for_kl_ref, prediction_post

        posterior_for_kl_ref = manager.get_best_place_hypothesis()['post'][:]
        prediction_post = posterior_for_kl_ref
            
        return posterior_for_kl_ref, prediction_post

#====================== Door Limit logic ==========================================================
    def initialise_place_doors_knowledge(self, manager:object) -> dict:
        """
        We retrieve all the doors present in memory for this place 
        and organise all the details to know if we crossed this door 
        Each entry of the dict contains:
        door_pose
        motion_axis
        direction
        origin_place
        connected_place
        exp_connected_place
        connected_place_door_pose
        """
        imagined_doors_info = {}
        current_exp_door_poses = manager.get_location_limits()
        for pose in current_exp_door_poses:
            imagined_doors_info = self.new_entry_in_imagined_doors_info(manager, imagined_doors_info, pose)
        return imagined_doors_info
    
    def new_entry_in_imagined_doors_info(self, manager:object, imagined_doors_info:dict, pose:list)-> dict:
        ''' create a door info entry in dict given a door pose, add it to dict'''
        pose_info = self.create_pose_limit_info_entry(manager,pose)
        new_door_entry = len(imagined_doors_info)
        imagined_doors_info[new_door_entry] = pose_info
        return imagined_doors_info
    
    def create_pose_limit_info_entry(self, manager:object, pose:list) -> dict:
        ''' Get all the infos necessary for place change given the door pose '''
        #initialise the entry dict
        door_info = call_process_limit_info(self.env_type,pose)
        #complete the entry dict
        door_info = self.door_info_update(manager, door_info)
        return door_info
    
    def door_info_update(self, manager, door_info):
        ''' 
        Add all we know about this door 
        (which place it leads to and through which door)
        '''
        #If we have yet to imagine what to expect passed this particular door
        posterior, expected_exp_id, place_door_pose = manager.get_connected_place_info(door_info['door_pose'])
        door_info['connected_place'] = posterior
        door_info['origin_place'] = manager.get_best_place_hypothesis()['post']
        door_info['exp_connected_place'] = expected_exp_id
        door_info['connected_place_door_pose'] = place_door_pose #not usefull to keep, but may be useful in later code changes
        return door_info

#====================== STEP BY STEP PREDICTION =====================================================

    def path_predicted_info_gain(self, manager:object, policy:list, imagined_doors_info:dict):
        """
        Adapted to Minigrid maze environment with doors and such,
        Get the KL of the prediction considering passing from place to place
        """

        #VISUALISATION TEST
        # predicted_img_seq = None

        #==== INIT ====#
        changing_place = -1
        path_steps_kl = []
        sequential_pose = manager.get_best_place_hypothesis()['pose'].copy()
        saved_current_place_pose = []
        info_printed_once = False

        for action in policy:
            sequential_pose = action_to_pose(action, sequential_pose)
            if existing_an_alternative_pose(saved_current_place_pose):
                saved_current_place_pose = action_to_pose(action, saved_current_place_pose)
            
            #Verify if the action leads to another place
            changing_place, sequential_pose, saved_current_place_pose = \
                self.verify_changing_place(manager, imagined_doors_info, \
                                           changing_place, sequential_pose, saved_current_place_pose)
            
            if self.verbose and changing_place >= 0 and not info_printed_once:
                print('changing place with policy:', transform_policy_from_hot_encoded_to_str(policy.tolist()))
                info_printed_once = True
            
            #Setting the posterior we are getting the observation from and the posterior we compare to
            posterior_for_kl_ref, prediction_post = \
                self.setting_prediction_and_comparison_places(manager, changing_place, imagined_doors_info)

            predicted_step = manager.single_pose_allocentric_prediction(sequential_pose, prediction_post)
                                           
            #--- Verify if we predict a door_view at this pose and if it's a new door  ---#
            imagined_doors_info = self.recognising_place_limit(manager, imagined_doors_info, predicted_step['image_predicted'], sequential_pose)

            #NOTE: reconstruction error has no meaning here, since it's pure prediction
            kl = float(calculate_KL(predicted_step.place, posterior_for_kl_ref).cpu().detach().numpy())   
            path_steps_kl.append(kl)
            
        #     if self.visualise_policy:
        #         #Do we want to see each policy predicted steps
        #         if predicted_img_seq is None:
        #             predicted_img_seq = predicted_step['image_predicted']
        #         else:
        #             predicted_img_seq = torch.cat((predicted_img_seq,predicted_step['image_predicted']),dim=1)
                         

        # if self.visualise_policy:
        #     str_policy = transform_policy_from_hot_encoded_to_str(policy)
        #     #if str_policy[0] == 'F':
        #         #Only watch the policies starting by going forward, to adapt at wish
        #     print('for policy', self.test_id, str_policy)
        #     print('G:', np.mean(path_steps_kl), 'steps KL', path_steps_kl)
        #     print('_')
        #     visualise_image(predicted_img_seq,title= self.test_id, fig_id=self.test_id)
        #     self.test_id += 1
            

        return path_steps_kl, imagined_doors_info

#====================== Place Changing logic ========================================================
    def recognising_place_limit(self, manager:object, imagined_doors_info:dict, \
                                predicted_image:torch.Tensor, sequential_pose:list) -> dict:
        '''
        Verify if we predict a door_view at this pose and if it's a new door. If it is we add it to the door dict

        '''
        pose = list(sequential_pose)
        pose_at_door = is_agent_at_door_given_ob(manager,predicted_image, pose, sensitivity=0.18)
        if pose_at_door :
            known_door_poses = [list(info['door_pose']) for info in imagined_doors_info.values()]
            if list(pose) not in known_door_poses:
                print('new door pose entry in memory:', pose, 'searching corresponding place')
                imagined_doors_info = self.new_entry_in_imagined_doors_info(manager,imagined_doors_info, pose)

        return imagined_doors_info
    
    def verify_changing_place(self,manager:object, imagined_doors_info:dict, \
                              changing_place:int, sequential_pose:list, saved_current_place_pose:list) -> tuple[int, list, list]:
        '''
        For each known door position we check if we are (or have passed) the door step. 
        If we are changing place (we are at door step) we mark it an verify which pose to replace it with (known place: place POV pose;
        new place: reset pose)
        Below a detailed description of the whole code:
        '''
        """
        # for each door in place we have to check if we passed that door (thus changing pov) (line 409)
        # FIRST OF ALL If we are in the process of passing a door and we are not considering the correct door.
        # Pass until we are at that door
        # NOW, to check if we are indeed passing a door
        # For that we consider the direction the agent is taking vs the direction it needs
        #       to take to pass that door, as well as which axis the agent needs to move on 
        #       (basically X or Y and - or + motion)
        #       the linear_motion is the value holding the direction of the correct axis
        # Thus we check if the sign of the linear motion is the exact same as the motion needed to pass the door
        # IF IT IS 
        # Have we passed the door and not facing back toward origin place? 
        # For that we check if we have passed the door AND we should not face backward 
        #          to it (face original place instead)
        # IFEVERYTHING HOLD TRUE up to now 
        # then we know we are on the verge of passing a door and which one. so we mark it
        # doors are not linked to exp id in memory, as such we previosuly inferred how both places are connected
        # so we need to know for the prediction: the other place and the door that links the 2 
        # (to get relative pose pov of other room)
        # THEN, have we passed the door for the 1st time ? 
        # we replace the current place pose with other place pose
        # New place: pose reset
        # known place: other room pose
        # we save the current room pose as well in case we go back to it
        # NOTE: since the door pose is the pose where we can see the closed door... 
        # we know the actual doorstep is 2 steps away.
        # NOTE: Since we change room reference the orientation we had in previous reference does not holds true anymore
        # thus we consider the global reference to get the position in the new room coordinate and have the orientation
        #//COMMENTED PART we check if we are not facing the door
        #if we don't, then we are not passing the door
        #//
        # Finally we check if, for this door, we are not facing opposite to the door after having switched pose pov
        #If so then we switch back to prev/current room pov again and mark that we are NOT passing a door 
        # if we are passing a door, then don't check the other doors, the work is done

        # with this we finished the door passing logic
        
        """
        for key, door_info in imagined_doors_info.items():
            #speed up process. we assume we can only pass 1 door by policy 
            # + it avoids issues if we reset the pose and it matches another door pose
            if changing_place >=0 and key != changing_place :
                continue
            index = door_info['motion_axis']
            linear_motion = sequential_pose[index] - door_info['door_pose'][index]
                            
            #If we are in front of door or at/passed door step 
            # //passed door view pose on the correct axis and direction
            if np.sign(linear_motion) == door_info['direction']:
                # If we are at least at door step AND we are not facing the current room
                #NOTE: when we are just facing open door, we do not change place of reference 
                # (that is a change from test phase)
                if passed_limit(linear_motion) and not backfacing_limit(sequential_pose,door_info['door_pose']):
                    changing_place = key
                    print('In imagination, changing place at pose:', sequential_pose, ', door info:', \
                        door_info['door_pose'], 'exp_connected_place:', door_info['exp_connected_place'],\
                        'connected_place_door_pose:', door_info['connected_place_door_pose'])
                    #print('\n'.join(f"{key}: {value}" for key, value in door_info.items() if key not in \
                    #               ['connected_place', 'origin_place', 'motion_axis', 'direction']))
                    #is_it_first_step_out_of_place ?
                    if not existing_an_alternative_pose(saved_current_place_pose):
                        #saving original place pose pov, in case we turn back to it
                        saved_current_place_pose = sequential_pose.copy()
                        sequential_pose = self.creating_new_pov_pose(manager, door_info, sequential_pose)
                        print('Changing place, thus switching policy pose from', saved_current_place_pose, 'to ', sequential_pose)
                
                #If we have changed place yet not passed front door or facing toward current room 
                # elif changing_place >= 0:
                #     changing_place = -1      
            
            #IF we have changed poc from original place accumulated_pose to new place pose
            # with THIS door view 
            # and we are turning back toward known room (180deg from door pose view)
            if key == changing_place and existing_an_alternative_pose(saved_current_place_pose) and backfacing_limit(saved_current_place_pose,door_info['door_pose']):
                #print('we are turning back on known room')
                sequential_pose = saved_current_place_pose
                saved_current_place_pose = []
                changing_place = -1
                break

            if changing_place != -1:
                break #we don't need to search through other door poses
        return changing_place, sequential_pose, saved_current_place_pose

    def creating_new_pov_pose(self,manager:object, door_info:dict, sequential_pose:list):
        """
        If we change place, we have to change local frame, 
        thus we have to adapt the pose used for place prediction
        """
        #if we have a door pose reference to set a new pose at door
        if len(door_info['connected_place_door_pose']) > 0:
            #We get the door pose from the pose of the door view
            forward_pose = from_door_view_to_door_pose(door_info['connected_place_door_pose'].copy())
            #We convert the current place orientation to the new place ref frame orientation
            forward_pose[2] = manager.convert_orientation_between_two_places_ref_frame(sequential_pose, goal_exp_id = door_info['exp_connected_place'])
            sequential_pose = forward_pose
        else:
            sequential_pose = np.array([0,0,0])
        
        return sequential_pose


#====================== CONDITIONS METHODS ==========================================================
def passed_limit(linear_motion:int)->bool:
    return abs(linear_motion) > 1
    
def existing_an_alternative_pose(saved_current_place_pose:list)-> bool:
    return len(saved_current_place_pose) > 0

def facing_limit(sequential_pose:list, door_pose:list)-> bool:
    return sequential_pose[2] == door_pose[2]

def backfacing_limit(sequential_pose:list, door_pose:list)-> bool:
    return sequential_pose[2] == (door_pose[2]+2)%4

def known_connected_place(exp_connected_place):
    return exp_connected_place is not None

