#!/usr/bin/env python3
#TESTS
import concurrent.futures

import gym
import gym_minigrid 

import matplotlib.pyplot as plt
import numpy as np
from gym_minigrid.wrappers import ImgActionObsWrapper, RGBImgPartialObsWrapper
from gym_minigrid.window import Window

from control_eval.input_output import *
from env_specifics.minigrid_maze_wt_aisles_doors.minigrid_maze_modules import (
    is_agent_at_door, set_door_view_observation)
from navigation_model.Processes.AIF_modules import mse_elements
from navigation_model.Processes.exploitative_behaviour import \
    Goal_seeking_Minigrid
from navigation_model.Processes.explorative_behaviour import \
    Exploration_Minigrid
from navigation_model.Processes.manager import Manager
from navigation_model.Services.model_modules import no_vel_no_action
from navigation_model.visualisation_tools import (
    convert_tensor_to_matplolib_list, transform_policy_from_hot_encoded_to_str,
    visualise_image)


def print_keys_explanation():
    print('NOTE: the agent needs a first push forward to initialise the model with a first observation. \
          So please press up arrow at least once to start well')
    print('==========KEYBOARD MANUAL===========')
    print('a: allow lazy user to launch an autonomous set of motion pre-defined in code')
    print('g: the current pose and observations are set as goal')
    print('r: agent seeks a goal (all the find+reach goal process)')
    print('t: enter current place pose and obtain the agent prediction (you will obtain it at next step)')
    print('d: agent apply step done -which means no motion, however agent still process observations')
    print('enter: agent explore')
    print('s: save current memory map')
    print('right or left arrows: agent goes left or right in its reference frame')
    print('up arrow: agent goes forward in its reference frame')
    print('====================================')


class MinigridInteraction():
    def __init__(self, args, redraw_window:bool = True) -> None:
        env_type = 'Minigrid_maze_aisle_wt_doors' + args.env
        self.redraw_window = redraw_window

        #AUTO TEST 
        f_move_aisle = ['Key.up']*6
        enter_keys = ['Key.enter']*80
        self.auto_motion = f_move_aisle + ['Key.left'] + f_move_aisle + ['Key.up']*2 + ['Key.left']*1 + f_move_aisle + ['Key.up']*3 + ['Key.right']
        self.auto_move = False
        self.goal_test = []
       
        #TODO: ALLOW USER TO SET ITS PREFERRED OB WTOUT ACCESSING CODE
        #Since 255 is the max value of any colour, we don't care about the above value of white
        self.preferred_colour_range = np.array([[235,235,235], [275,275,275]])

        # Minigrid HOT encoded actions
        forward = [1,0,0]
        right = [0,1,0]
        left = [0,0,1]
        mingrid_actions = [forward, right, left]

        #--- MODEL INIT ---#
        allo_config = setup_allocentric_config(args.allo_config)
        memory_config = setup_memory_config(args.memory_config)
        old_memory = args.memory_load

        self.models_manager = Manager(allo_config, memory_config, mingrid_actions, env= env_type, lookahead=args.lookahead)
        set_door_view_observation(self.models_manager)
        load_memory(self.models_manager,old_memory)

        #--- explorative_behaviour INIT ---#
        self.explorative_behaviour = Exploration_Minigrid(env_type, possible_actions = mingrid_actions, curiosity_temp= 100)

        self.exploitative_behaviour = Goal_seeking_Minigrid(env_type)

        #=== Tests variables ===#
        Minigrid_env_details = args.env[0] +'t_'+ str(args.rooms_in_row)+'x'+str(args.rooms_in_col) + '_s'+ str(args.seed)
        saving_dir = create_saving_directory(args.save_dir)
        self.saving_dir = saving_dir + '/' + Minigrid_env_details  
        create_directory(self.saving_dir) 
        Minigrid_env_details
        self.automatic_process = False

        #--- ENV INIT ---#
        self.env_name = 'MiniGrid-' + args.env + '-v0'
        print(self.env_name)
        self.env = gym.make(self.env_name, rooms_in_row=args.rooms_in_row, rooms_in_col=args.rooms_in_col)
        self.env = RGBImgPartialObsWrapper(self.env)
        self.env = ImgActionObsWrapper(self.env)
        self.window = Window(self.env_name)
        self.seed = args.seed
        self.reset()

        self.windows_key_handler()

#==================== MINIGRID ENV METHODS ====================#
    def windows_key_handler(self)-> None:
        ''' 
        Allows keyboard handling 
        only if we have a window display
        '''
        if self.redraw_window:
            print_keys_explanation()
            self.window.reg_key_handler(self.key_handler)        
            # Blocking event loop
            self.window.show(block=True) 

    def redraw(self) -> list:
        '''
        get observation, return whole world image
        '''
        img = self.env.render('rgb_array', tile_size=32)
        if self.redraw_window:
            self.window.show_img(img)
        return img

    def reset(self)->None:
        if self.seed != -1:
            self.env.seed(self.seed)
        obs = self.env.reset()
        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.window.set_caption(self.env.mission)
        if self.redraw_window:
            self.redraw()

    def step_count(self)->int:
        return self.env.step_count + 1
    
    def agent_absolute_pose(self)->list:
        return [*self.env.agent_pos,self.env.agent_dir]

    def step(self,action)->bool:
        
        self.agent_step(action)
        self.redraw()

        print('____________')
        print()
        if self.auto_move:
            try:
                del self.auto_motion[0]
                if len(self.auto_motion) > 0 :
                    press_keyboard(self.auto_motion[0])
                else:
                    print('end of auto motion')
            except IndexError:
                pass
        return self.models_manager.agent_lost()

    def convert_hot_encoded_to_minigrid_action(self, action:list) -> int:
        if len(action) < 1 :
            print('No action, not moving')
            return self.env.actions.done
        if action[0] == 1:
                action = self.env.actions.forward
        elif action[1] == 1:
            action = self.env.actions.right
        elif action[2] == 1:
            action = self.env.actions.left
        else:
            raise Exception('unrecognised action to apply:'+str(action))
        return action
        
    def key_handler(self,event:str) -> None:
        print('pressed', event.key)
        key = event.key

        if key == 'a':
            print('automotion')
            self.auto_move = True
            press_keyboard(self.auto_motion[0])
        
        if key == 'g':
            print('this pose and ob is the new goal')
            obs, _, _, _ = self.env.step(self.env.actions.done)
            self.goal_test = self.exploitative_behaviour.set_current_pose_as_goal(self.models_manager, obs['image'])

        if key == 'r':
            print('Agent seeks the goal, it will search for preferred colour range in pred + apply policy to reach it')
            self.agent_seeks_goal()
            if self.automatic_process:
                press_keyboard()
            return

        if key == 't':
            #VISUALISATION PREDICTION FOR A GIVEN POSE
            print('Enter coordinates x,y,theta of the pose \
                  from which you want the allo model img prediction')
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_user_input)
                user_pose = future.result()
            print("You want to know the predicted image for pose:", user_pose)
            pred_ob = self.models_manager.given_pose_prediction_visualisation(user_pose)
            obs, _, _, _ = self.env.step(self.env.actions.done)
            obs = self.models_manager.allocentric_process.allocentric_model.torch_observations(obs)
            visualise_image(obs['image'].unsqueeze(0), 'GT pose:'+ str(user_pose), fig_id = 101)
            obs_image = obs['image'].unsqueeze(0).unsqueeze(0).repeat(pred_ob.shape[0], 1,1,1,1)
            print(obs_image.shape, pred_ob.shape)
            print('MSE PRED', mse_elements(pred_ob, obs_image))
            print('_________')
            print()
        
        if key == 'd':
            print('Next step, do nothing')
            self.step(self.env.actions.done)
         
        if key == 'p':
            #Erase all plots from 10 to 300
            print('closing all plots rangin from 10 to 300')
            for id in range(10,1500):
                plt.close(id)
        if key == 'enter':
        #explorative_behaviour
            self.agent_explores()

            if self.automatic_process:
                press_keyboard()
            return

        #Manual keys:
        if key == 'left':
            self.step(self.env.actions.left)
            if self.automatic_process:
                press_keyboard()
            return 
        
        if key == 'right':
            self.step(self.env.actions.right)
            if self.automatic_process:
                press_keyboard()
            return 
        
        if key == 'up':
            self.step(self.env.actions.forward)
            if self.automatic_process:
                press_keyboard()
            return 
        if key == 's':
            save_memory(self.models_manager, self.saving_dir)
            return 

#==================== MODEL METHODS ====================#
    def agent_lost(self)->bool:
        return self.models_manager.agent_lost()
    
    def agent_situate_memory(self)->int:
        return self.models_manager.get_current_exp_id()

    def agent_step(self, action) -> tuple[bool,dict]:
        print('step in world:', self.step_count())
        print('action to apply:',action)
        obs, _, _, _ = self.env.step(action)
        print(obs.keys(),obs["pose"])
        obs = no_vel_no_action(obs)
        self.models_manager.digest(obs)
        is_agent_at_door(self.models_manager, sensitivity=0.72)
        print("is agent at door",is_agent_at_door(self.models_manager, sensitivity=0.54))
        if self.models_manager.agent_lost():
            #we are lost? want the info_gain memory to be reset to 1 value
            slide_window = 1
        else:
            slide_window = 5
        info_gain = self.models_manager.get_best_place_hypothesis()['info_gain']
        self.explorative_behaviour.update_rolling_info_gain(info_gain, window=slide_window)
        print('Step Done')
        return self.models_manager.agent_lost(), obs
    
    def apply_policy(self, policy:list, n_actions:int, collect_data:bool=False)->tuple[list,bool]:
        motion_data = []
        agent_lost = False
        print("n_sctions",n_actions, policy,self.convert_hot_encoded_to_minigrid_action(policy[0]))
        #for a in range(len(policy)):
        for a in range(n_actions):
            action = self.convert_hot_encoded_to_minigrid_action(policy[a])
            agent_lost, obs = self.agent_step(action)
            world_img = self.redraw()
            if collect_data:
                data = self.collect_data_models()
                data['env_image'] = world_img
                data['ground_truth_ob'] = obs['image']
                motion_data.append(data)
        
            if agent_lost:
                print('Agent lost, stopping exploration policy')
                return motion_data, agent_lost
        return motion_data, agent_lost
    
    def agent_explores(self, collect_data:bool=False)->tuple[list,bool]:
        
        policy, n_actions = self.apply_exploration()
        motion_data,agent_lost = self.apply_policy(policy, n_actions, collect_data)
        return motion_data, agent_lost

    def apply_exploration(self)->tuple[list,int]:
        print('====================================================explo')
        if self.models_manager.agent_lost():
            print('Agent lost, trying to determine best place hypothesis')
            policy, n_action = self.explorative_behaviour.solve_doubt_over_place(self.models_manager)
            print("solve doubt over place", policy, n_action)
            return policy, n_action
        
        ongoing_exploration_option = 'explore'
        print('exploring')
        if self.explorative_behaviour.is_agent_exploring(): #agent_exploring return the latest computed value
            policy, n_action = self.explorative_behaviour.one_step_ego_allo_exploration(self.models_manager)
            print("is agent exploring?",self.explorative_behaviour.is_agent_exploring())
            print(policy, n_action )
        else:
            policy, n_action = [],[]
        
              
        while len(policy) == 0:    
            print('++++++++++++++++++++++++ while len policy =0')   
            print('trying to increase lookahead by 1 to increase exploration range')
            lookahead_increased = self.models_manager.increase_lookahead(max=7)   
            if not lookahead_increased:
                print('lookahead at max value, search to return to another place')
                ongoing_exploration_option = 'change_memory_place'
              
            if ongoing_exploration_option == 'change_memory_place':
                print('Searching for another place to go to')
                place_to_go = self.models_manager.connected_place_to_visit()  
                print('++++++++++++++++++++++++') 
                if place_to_go:
                    print('Searching How to go to the other place')
                    policy, n_action = self.exploitative_behaviour.go_to_given_place(\
                    self.models_manager,place_to_go['current_exp_door_pose'])
                else:
                    print('No place found to return to, using ego model')
                    ongoing_exploration_option = 'push_from_comfort_zone'
                
            elif ongoing_exploration_option == 'explore':
                print('Exploring with allo model')
                policy, n_action = self.explorative_behaviour.one_step_ego_allo_exploration(self.models_manager)

            if ongoing_exploration_option == 'push_from_comfort_zone':
                print('Using ego model to push exploration')
                policy, n_action = self.explorative_behaviour.one_step_egocentric_exploration(self.models_manager)

        print('++++++++0 policy should be before this++++++++++++++++++++++++') 
        policy_G = self.explorative_behaviour.get_latest_policy_EFE(reset_EFE=True)
        print("policyG",policy_G)
         #we get policy_G and then erase it from memory
        if policy_G is not None: 
            info_gain_coeff = self.explorative_behaviour.rolling_coeff_info_gain()
            print("info gain coeff", info_gain_coeff)
            if self.explorative_behaviour.define_is_agent_exploring(info_gain_coeff, policy_G, threshold=1):
                self.models_manager.reset_variable_lookahead_to_default()
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')            
        return policy, n_action

    def agent_seeks_goal(self, collect_data:bool=False)->tuple[list,bool]:
        motion_data, agent_lost = self.apply_goal_seeking(collect_data)
        goal = self.exploitative_behaviour.get_preferred_objective()
        if 'image' in goal and goal['image'] is not None:
            for step_data in motion_data:
                step_data['goal'] = goal['image']
   
        print('APPLIED REACH OBJECTIVE POLICY')
        return motion_data, agent_lost
    
    def apply_goal_seeking(self, collect_data:bool=False)->tuple[list,bool]:
        
        if self.models_manager.agent_lost():
            return self.agent_explores(collect_data)
            #DONE
        goal = self.exploitative_behaviour.get_preferred_objective()
        #DO THIS RESEARCH EVERY FEW POLICIES
        if not goal or goal['decay'] <= 0 :
            print('=================================== SEARCH GOAL ==========================')
            goal = self.exploitative_behaviour.define_preferred_objective(self.models_manager, self.preferred_colour_range)
        #TMP TEST VISUALISATION
        # if goal and goal['image'] is not None:
        #     if len(goal['image'].shape)<4:
        #         img = goal['image'].unsqueeze(0)
        #     else:
        #         img = goal['image']
        #     visualise_image(img, title='goal', fig_id=11)
        if not goal :
            print('=================================== NO GOALFOUND TO REACH ==========================')
            return self.agent_explores(collect_data)
            #DONE
        # elif 'pose' in goal and goal['pose'][:2] == self.models_manager.get_best_place_hypothesis()['pose'][:2]:
        #     #We are on the preferrence, nothing to do
        #     return {}, False
        
        elif goal['exp'] == self.models_manager.get_current_exp_id():
            print('=================================== GOAL FOUND IN PLACE ==========================')
            policy, error = self.exploitative_behaviour.go_to_observation_in_place(self.models_manager, goal.copy())
            if not error:
                if len(policy) > 0:
                    n_actions = 1
                else:
                    n_actions = 0
                
            else:
                #We failed to reach goal, we will retry process
                logging.warning('WARNING: apply_goal_seeking- Error in go_to_observation_in_place with policy' + str(transform_policy_from_hot_encoded_to_str(policy)))
                return [], False
            #NOTE: THIS CAN RETURN ERROR IF EGO MODEL SEE GOAL IN EXP, BUT FURTHER THAN LOOKAHEAD
            #THEN ALLO CALLED, AND ALLO WRONG PLACE REPRESENTATION. Anyway, likely a one time error pred as it happens.

        else:
            print('=================================== REACH EXP CONTAINING GOAL s==========================')
            path = self.exploitative_behaviour.apply_djikstra_over_memory(self.models_manager, goal)
            place_to_go = self.exploitative_behaviour.determine_location_to_reach_in_path(self.models_manager,path)
            print('Chosen place to go to reach objetive exp', place_to_go)
            if place_to_go:
                policy, n_actions = self.exploitative_behaviour.go_to_given_place(\
                    self.models_manager,place_to_go['current_exp_door_pose'])
            else:
                print('This place is not connected to any we know to lead toward the goal.')
                return self.agent_explores(collect_data)
        #The memory of the goal decay over time when reaching it, 
        # thus requiring refreshing of memory from time to time
        self.exploitative_behaviour.goal_memory_decay()
        motion_data,agent_lost = self.apply_policy(policy, n_actions, collect_data)
        return motion_data,agent_lost 
        
    def collect_data_models(self)->dict:
        """collect latest data from manager """
        ''' this is for visualisation purposes only '''
        data = {}
        place_description = self.models_manager.get_best_place_hypothesis()
        data.update(place_description)
        if 'post' in data:
            del data['post']

        if 'image_predicted' in data and data['image_predicted'] is not None:
        #convert to numpy for data visualisation
            data['image_predicted'] = convert_tensor_to_matplolib_list(data['image_predicted'])

        #----- just PLOT DATA -----#
        data['GP'] = self.models_manager.memory_graph.get_global_position() 
        data['exp'] = self.models_manager.memory_graph.get_current_exp_id()

        if 'mse' not in data :
            data['mse'] = -1
        if 'kl' not in data :
            data['kl'] = 0
        if 'info_gain' not in data :
            data['info_gain'] = 0
        #TEST
        # exp_ids, probs = self.exp_visualisation_proba(place_descriptors)
        # data['exp_ids'] = [exp_ids]
        # data['exp_prob'] = [probs]

        return data
    
    def save_created_map(self, saving_dir:str)->None:
        save_memory(self.models_manager, saving_dir)
    #TODO: refactor
    def get_memory_map_data(self):
        ''' 
        This is for plotting the memory map
        
        TO DO: REFACTOR'''

        memory_map_data = {'exps_GP':[], 'exps_decay': [], 'ghost_exps_GP':[],\
                           'ghost_exps_link':[], 'exps_links': []}
        memory_map_data['current_exp_id'] = self.models_manager.memory_graph.get_current_exp_id()
        memory_map_data['current_GP'] = self.models_manager.memory_graph.get_global_position()
        if memory_map_data['current_exp_id'] < 0:
            return memory_map_data
        
        current_exp_GP = self.models_manager.memory_graph.get_exp_global_position() 
        memory_map_data['current_exp_GP'] = current_exp_GP
        #GET EXPS POSITIONS
        for vc in self.models_manager.memory_graph.view_cells.cells:
            for exp in vc.exps:
                memory_map_data['exps_GP'].append([exp.x_m, exp.y_m])
                memory_map_data['exps_decay'].append(vc.decay)

        #GET GHOST EXPS POSITIONS
        for expc in self.models_manager.memory_graph.experience_map.ghost_exps:
            memory_map_data['ghost_exps_GP'].append([expc.x_m, expc.y_m])
            #GET GHOST NODES LINKS
            for l_id in range(0,len(expc.links)):
                memory_map_data['ghost_exps_link'].append([expc.x_m, expc.y_m])
                memory_map_data['ghost_exps_link'].append([expc.links[l_id].target.x_m, expc.links[l_id].target.y_m])
        #GET EXPS LINKS      
        for expc in self.models_manager.memory_graph.experience_map.exps:
                if len(expc.links)>0:
                    #print('in visualisation tool', expc.id, len(expc.links))
                    for l_id in range(0,len(expc.links)): 
                        if expc.links[l_id].target.ghost_exp == False:
                            #print('target id', expc.links[l_id].target.id)
                            memory_map_data['exps_links'].append([expc.links[l_id].target.x_m,expc.links[l_id].target.y_m])
                            memory_map_data['exps_links'].append([expc.x_m, expc.y_m])
                       
        return memory_map_data




#-----------__MAIN -----------------------------------------------------------------------------------------------------------
def main():
    try:
        args = parser.parse_args()
        test = MinigridInteraction(args)

    finally:
        pass

if __name__ == "__main__":
# instantiating the decorator
    main()
  