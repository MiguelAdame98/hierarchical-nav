#!/usr/bin/env python3
#TESTS
import concurrent.futures
from typing import List, Dict, Any
from collections import defaultdict

import gym
import gym_minigrid 
#import rospy
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

import matplotlib.pyplot as plt
import numpy as np
import random
from nltk import PCFG
from nltk.parse.generate import generate
from gym_minigrid.wrappers import ImgActionObsWrapper, RGBImgPartialObsWrapper
from gym_minigrid.window import Window
from collections import deque

from control_eval.input_output import *
from collections import Counter
from navigation_model.Processes.motion_path_modules import action_to_pose

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
    visualise_image,visualize_replay_buffer)
from copy import deepcopy


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
        self.replay_buffer = []  # List to store recent state dictionaries.
        self.buffer_size = 600
       
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
    def convert_minigrid_action_to_hot_encoded(self, action: int) -> list:
        if action == self.env.actions.forward:
            return [1, 0, 0]
        elif action == self.env.actions.right:
            return [0, 1, 0]
        elif action == self.env.actions.left:
            return [0, 0, 1]
        else:
            raise Exception("Unrecognized minigrid action: " + str(action))
    def convert_list_to_hot_encoded(self, actions_list: list) -> list:
        # Initialize an empty list for the one-hot encoded actions.
        hot_encoded_actions = []

        # Iterate over each action in the given list.
        for action in actions_list:
            # Use your conversion function to convert the action into one-hot encoded format.
            encoded_action = self.convert_minigrid_action_to_hot_encoded(action)
            hot_encoded_actions.append(encoded_action)

        return hot_encoded_actions    
        
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
    
    def update_replay_buffer(self, state):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)  # Remove the oldest entry to keep the size fixed.
        self.replay_buffer.append(deepcopy(state))

    def extract_past_actions(self,replay_buffer):
        actions = []
        for state in replay_buffer:
            # Get the action from the state; use None as the default if not found.
            action = state.get('action', None)
            if action is not None:
                actions.append(action)
        return actions
    def extract_past_real_poses(self,replay_buffer):
        actions = []
        for state in replay_buffer:
            # Get the action from the state; use None as the default if not found.
            action = state.get('real_pose', None)
            if action is not None:
                actions.append(action)
        return actions
    def extract_past_poses(self,replay_buffer):
        actions = []
        for state in replay_buffer:
            # Get the action from the state; use None as the default if not found.
            action = state.get('imagined_pose', None)
            if action is not None:
                actions.append(action)
        return actions
    def action_to_past_pose(self, action, final_pose):
        DIR_TO_VEC = [
            [1, 0],    # 0: pointing right (positive X)
            [0, 1],    # 1: down (positive Y)
            [-1, 0],   # 2: left (negative X)
            [0, -1]    # 3: up (negative Y)
        ]

        initial_pose = final_pose.copy()
        if action[0] == 1:
            print("action[0] == 1:")
            initial_pose[0] = final_pose[0] - DIR_TO_VEC[final_pose[2]][0]
            initial_pose[1] = final_pose[1] - DIR_TO_VEC[final_pose[2]][1]
        elif action[0] == -1:
            initial_pose[0] = final_pose[0] + DIR_TO_VEC[final_pose[2]][0]
            initial_pose[1] = final_pose[1] + DIR_TO_VEC[final_pose[2]][1]
        elif action[1] == 1:
            print("action[1] == 1:")
            initial_pose[2] = (final_pose[2] - 1) % 4
        elif action[2] == 1:
            print("action[2] == 1:")
            initial_pose[2] = (final_pose[2] + 1) % 4

        return initial_pose
    def extract_branch_paths(self, tree, current_path=None):
        if current_path is None:
            current_path = {"actions": [], "poses": [tree["pose"]]}
            #print(f">>> start at pose={tree['pose']}")

        kids = tree.get("children") or []
        if not kids:
            # We’re at a leaf
            L = len(current_path["actions"])
            #print(f"   leaf @ pose={tree['pose']} ➞ length={L}")
            return [current_path]

        paths = []
        for child in kids:
            action, pose = child["action"], child["pose"]
            new_path = {
                "actions": current_path["actions"] + [action],
                "poses":   current_path["poses"]   + [pose]
            }
            print(f"  desc → action {action} ➞ pose {pose}  (so far: {len(new_path['actions'])} actions)")
            paths.extend(self.extract_branch_paths(child, new_path))

        return paths
    def build_decision_tree_all(self, current_pose, depth, max_depth, branch_history=None):
        # Initialize the branch history if necessary
        if branch_history is None:
            branch_history = [current_pose]
        
        # Create the current node.
        node = {
            "pose": current_pose,
            "action": None,   # Root node doesn't have an action that produced it.
            "children": [],
            "depth": depth
        }
        
        # Stop expansion if maximum depth is reached.
        if depth >= max_depth:
            return node

        # Define the available actions in your grid world.
        AVAILABLE_ACTIONS = [[1,0,0], [0,1,0], [0,0,1]]
        
        # For each action, simulate the new pose.
        for action in AVAILABLE_ACTIONS:
            # Compute new_pose using your forward update function.
            new_pose = action_to_pose(np.array(action), current_pose.copy())
            if any(np.array_equal(new_pose, p) for p in branch_history):    
                continue

            new_branch_history = branch_history + [new_pose]
            
            subtree = self.build_decision_tree_all(new_pose, depth + 1, max_depth, new_branch_history)
        
            # Create a child node that includes the action that led to new_pose.
            child_node = {
                "pose": new_pose,
                "action": action,   # Store this action here.
                "children": subtree["children"],  # Adopt the children from the subtree.
                "depth": depth + 1
            }
            # Optionally, include any additional fields from subtree into child_node.
            node["children"].append(child_node)
            
        return node
    def _recency_weight(self,age, tau, kind="exp"):
        """age = 0 means ‘just visited’.  Larger age → smaller weight."""
        if kind == "exp":            # exponential decay
            return np.exp(-age / tau)
        elif kind == "hyper":        # hyperbolic (1 / (1+age))
            return 1.0 / (1.0 + age)
        else:                        # constant (no decay)
            return 1.0
    def _rollout_poses(self, start_pose: np.ndarray,
                   actions: np.ndarray,
                   action_to_pose) -> List[np.ndarray]:
        """Re‑compute poses after the policy was shortened by collision trimming."""
        poses = [start_pose.copy()]
        cur   = start_pose.copy()
        for a in actions:
            cur = action_to_pose(a, cur.copy())
            poses.append(cur)
        return poses   

    
    def evaluate_branches(
    self,
    branches: List[Dict[str, Any]],     # [{actions, poses}, …] from extract_branch_paths
    past_poses: List[np.ndarray],
    tau: float = 50,
    decay: str = "hyper",
    *,
    eps: float = 0.5,
    max_keep: int | None = None,
    embed_fn=None
):
        """
        1. Feed **action sequences only** to egocentric collision check.
        2. Reconstruct pose sequences for the surviving (trimmed) policies.
        3. Score with recency penalty.
        4. Apply radius‑greedy diversity filter.
        5. Return list of (penalty, actions, poses) tuples OR just actions.
        """
        # ---- 0. default embedding on final (x,y,θ) pose ----
        if embed_fn is None:
            embed_fn = lambda poses: np.asarray(poses[-1], dtype=float)

        # ---- 1. collision + dedup first ----
        action_lists = [br["actions"] for br in branches]
        pruned_pols  = self.models_manager.get_plausible_policies(action_lists)
        #           -> list[torch.Tensor]   each shape (T,3)

        # ---- 2. rebuild pose traces so we can score ----
        # Build quick lookup: initial pose for each ORIGINAL branch
        # (all branches share the same start pose in typical tree; otherwise
        #  keep a parallel list or include start_pose in each dict)
        start_pose = branches[0]["poses"][0]

        rebuilt = []   # list of dicts with keys actions, poses
        for pol in pruned_pols:
            acts  = pol.cpu().numpy()              # (T,3)
            poses = self._rollout_poses(start_pose, acts, action_to_pose)
            rebuilt.append({"actions": acts, "poses": poses})

        # ---- 3. recency‑penalty scoring ----
        last_seen = {tuple(p[:2]): t            # 👈  keep only x,y !
             for t, p in enumerate(past_poses)}
        now = len(past_poses)
        current_key = tuple(start_pose[:2])     # (x0, y0)
        last_seen[current_key] = now 

        scored = []
        for br in rebuilt:
            penalty = 0.0
            for pose in br["poses"]:
                key = tuple(pose[:2])           # 👈  ignore θ during lookup
                if key in last_seen:
                    age     = now - last_seen[key]
                    penalty += self._recency_weight(age, tau, decay)
            scored.append((penalty, br))
        current_xy = np.asarray(start_pose[:2], dtype=float)
        scored.sort(key=lambda x: (x[0],              # 1️⃣ low penalty
                           np.linalg.norm(embed_fn(x[1]["poses"])[:2] - current_xy)))

        ALPHA = 60.0         # weight for recency‑penalty
        BETA  = 0.5         # weight for outward distance (tweak!)

        best_branch = None
        best_score  = np.inf

        for pen, br in scored:                       # kept comes out of radius‑greedy
            end_xy = np.asarray(br["poses"][-1][:2], dtype=float)
            dist   = np.linalg.norm(end_xy - current_xy)      # outward distance
            lin    = ALPHA * pen - BETA * dist               # lower is better
            print(pen,br)
            print("who is winning",ALPHA * pen,BETA * dist,)
            print(f"lin={lin:.10f}   best_score={best_score:.10f}")
            if lin < best_score:
                print("inside the best score",lin,br)
                best_score  = lin
                best_branch = br

        # what you return / store
        return_best_actions = torch.as_tensor(best_branch["actions"])
        
        return return_best_actions
        

    def compute_pose_history(self):
        # If the replay buffer is empty, return an empty list.
        if not self.replay_buffer:
            return []
        
        # Get the initial pose from the first state.
        first_state = self.replay_buffer[-1]
        if 'imagined_pose' in first_state and first_state['imagined_pose'] is not None:
            current_pose = list(first_state['imagined_pose'])
        else:
            current_pose = list(first_state.get('real_pose', [0, 0, 0]))  # assume pose format: [x, y, theta]
        
        pose_history = [current_pose.copy()]
        #print(pose_history)
        # Iterate through the replay buffer in forward order.
        # For each state, update the current pose using the stored action.
        for state in reversed(self.replay_buffer):
            action = state.get('action')
            hotenc_action = np.array(self.convert_minigrid_action_to_hot_encoded(action), dtype=np.int32)
            #print("Hot-encoded action:", hotenc_action, type(hotenc_action), hotenc_action.shape)
            if action is None:
                # If no action is specified, assume no movement.
                continue
            # Compute the updated pose: this function should update the pose given the action.
            current_pose = self.action_to_past_pose(hotenc_action, current_pose.copy())
            #print("fff",current_pose)
            # Append a copy of the updated pose to the trajectory history.
            pose_history.append(current_pose.copy())
        
        return pose_history

    

    def agent_step(self, action) -> tuple[bool,dict]:
        print('step in world:', self.step_count())
        print('action to apply:',action)
        
        #bridge = CvBridge()
        #img_msg = rospy.wait_for_message("/camera/depth_registered/rgb/image_raw", Image, timeout=10)
        #img_bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        #print(img_msg)
        #print(bridge,img_msg,img_bgr.shape)
        obs, _, _, _ = self.env.step(action)
        print(obs.keys(),obs["pose"], obs["image"].shape)
        obs = no_vel_no_action(obs)
        self.models_manager.digest(obs)
        is_agent_at_door(self.models_manager,obs["image"], sensitivity=0.18)
        print("is agent at door",is_agent_at_door(self.models_manager,obs["image"], sensitivity=0.18))
        if self.models_manager.agent_lost():
            #we are lost? want the info_gain memory to be reset to 1 value
            slide_window = 1
        else:
            slide_window = 5
        info_gain = self.models_manager.get_best_place_hypothesis()['info_gain']
        self.explorative_behaviour.update_rolling_info_gain(info_gain, window=slide_window)
        current_state = {
        'real_pose': obs["pose"],
        'imagined_pose':self.models_manager.get_best_place_hypothesis()['pose'],
        'real_image': obs['image'],   
        'imagined_image': self.models_manager.get_best_place_hypothesis()['image_predicted'],
        'action':action}
        self.update_replay_buffer(current_state)
        if self.step_count()>20:
            grammar = self.build_pcfg_from_memory()
            plan = list(generate(grammar, n=1))[0]
            print("[PLAN] Sampled PCFG plan:", plan)

        print('Step Done')
        return self.models_manager.agent_lost(), obs
    
    def apply_policy(self, policy:list, n_actions:int, collect_data:bool=False)->tuple[list,bool]:
        motion_data = []
        agent_lost = False
        print("n_sctions",n_actions, policy,self.convert_hot_encoded_to_minigrid_action(policy[0]))
        for a in range(len(policy)):
        #for a in range(n_actions):
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
    
    def push_explo_stm(self):
            pose_history=self.compute_pose_history()
            self.extract_past_actions(self.replay_buffer)
            a=self.extract_past_poses(self.replay_buffer)
            b=self.extract_past_real_poses(self.replay_buffer)
            #print("nnn",a)
            #print(b)
            max_depth = 6
            tree_all = self.build_decision_tree_all(self.replay_buffer[-1]["imagined_pose"], 0, max_depth)
            '''def print_tree(node, indent=0):
                print(" " * indent, f"Depth: {node['depth']}, Pose: {node['pose']}, Action: {node['action']}")
                for child in node["children"]:
                    print_tree(child, indent + 2)'''
    
            paths=self.extract_branch_paths(tree_all)
            n_branches = len(paths)
            print(f"Number of branches: {n_branches}")
            actions_per_branch = [len(p['actions']) for p in paths]
            poses_per_branch   = [len(p['poses'])   for p in paths]

            # 3. Count how many branches fall into each length
            action_length_counts = Counter(actions_per_branch)
            pose_length_counts   = Counter(poses_per_branch)

            # 4. Print a nice summary
            print("Branches by # of actions:")
            for length, count in sorted(action_length_counts.items(), reverse=True):
                print(f"  {count} branch{'es' if count>1 else ''} with {length} action{'s' if length!=1 else ''}")

            print("\nBranches by # of poses:")
            for length, count in sorted(pose_length_counts.items(), reverse=True):
                print(f"  {count} branch{'es' if count>1 else ''} with {length} pose{'s' if length!=1 else ''}")
            top_diverse=self.evaluate_branches(paths,pose_history)
            print(top_diverse)
            print(top_diverse.tolist(),len(top_diverse))
            return top_diverse.tolist(),len(top_diverse)
    
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
            lookahead_increased = self.models_manager.increase_lookahead(max=5)   
            if not lookahead_increased:
                print('lookahead at max value, search to return to another place')
                ongoing_exploration_option = 'change_memory_place'
              
            if ongoing_exploration_option == 'change_memory_place':
                print('Searching for another place to go to')
                print("this time for real",self.step_count())
                policy, n_action= self.push_explo_stm()  
                print('this time ++++++++++++++++++++++++',self.step_count()) 
                '''if place_to_go:
                    print('Searching How to go to the other place')
                    print("where are we going?",place_to_go['current_exp_door_pose'],place_to_go)
                    policy, n_action = self.exploitative_behaviour.go_to_given_place(\
                    self.models_manager,place_to_go['current_exp_door_pose'])
                
                else:
                    print('No place found to return to, using ego model')
                    ongoing_exploration_option = 'push_from_comfort_zone'''
                
            elif ongoing_exploration_option == 'explore':
                print('Exploring with allo model')
                policy, n_action = self.explorative_behaviour.one_step_ego_allo_exploration(self.models_manager)

            if ongoing_exploration_option == 'push_from_comfort_zone':
                print('Using ego model to push exploration')
                policy, n_action = self.explorative_behaviour.one_step_egocentric_exploration(self.models_manager)

        print('++++++++0 policy should be before this++++++++++++++++++++++++') 
        policy_G = self.explorative_behaviour.get_latest_policy_EFE(reset_EFE=True)
        print("policyG",policy_G ,self.explorative_behaviour.is_agent_exploring())
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
    '''def get_memory_map_data(self):
        ''' 
    '''This is for plotting the memory map
        
        TO DO: REFACTOR''''''

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
                       
        return memory_map_data'''
    def get_memory_map_data(self, dbg=True):
        """
        This is for plotting the memory map.
        Adds debug dumps of every link and deduplicates before returning.
        """
        mg = self.models_manager.memory_graph
        emap = mg.experience_map

        # 1) Prepare the empty payload
        memory_map_data = {
            'exps_GP': [], 
            'exps_decay': [], 
            'ghost_exps_GP': [],
            'ghost_exps_link': [], 
            'exps_links': []
        }

        # 2) Current experience and global pose
        memory_map_data['current_exp_id'] = mg.get_current_exp_id()
        memory_map_data['current_GP']     = mg.get_global_position()
        if memory_map_data['current_exp_id'] < 0:
            if dbg:
                print("[DBG] No current experience → empty map")
            return memory_map_data

        memory_map_data['current_exp_GP'] = mg.get_exp_global_position()

        # 3) DEBUG: dump every existing link in the graph
        if dbg:
            print("[DBG] Full link graph (exp → targets):")
            for exp in emap.exps:
                targets = [link.target.id for link in exp.links]
                print(f"   Exp {exp.id} → {targets}")

        # 4) Collect experience nodes
        for vc in mg.view_cells.cells:
            for exp in vc.exps:
                memory_map_data['exps_GP'].append([exp.x_m, exp.y_m])
                memory_map_data['exps_decay'].append(vc.decay)

        # 5) Ghost experiences + their links
        for ghost in emap.ghost_exps:
            memory_map_data['ghost_exps_GP'].append([ghost.x_m, ghost.y_m])
            for link in ghost.links:
                memory_map_data['ghost_exps_link'].append([ghost.x_m, ghost.y_m])
                memory_map_data['ghost_exps_link'].append([link.target.x_m, link.target.y_m])

        # 6) Real experience‐to‐experience links
        for exp in emap.exps:
            for link in exp.links:
                # skip ghost‐to‐ghost or ghost‐to‐real if any slipped through
                if not getattr(link.target, 'ghost_exp', False):
                    memory_map_data['exps_links'].append([link.target.x_m, link.target.y_m])
                    memory_map_data['exps_links'].append([exp.x_m, exp.y_m])

        # 7) DEBUG: show raw link list
        if dbg:
            raw = memory_map_data['exps_links']
            n_pairs = len(raw) // 2
            print(f"[DBG] Raw exps_links pairs: {n_pairs}, raw list (first 6 points) = {raw[:6]}")

        # 8) Deduplicate accidental duplicates
        clean = []
        seen = set()
        pts = memory_map_data['exps_links']
        # interpret as consecutive pairs [p0, p1, p2, p3, ...]
        for i in range(0, len(pts), 2):
            p0 = tuple(pts[i])
            p1 = tuple(pts[i+1])
            pair = (p0, p1)
            if pair not in seen:
                seen.add(pair)
                clean.extend([list(p0), list(p1)])
        memory_map_data['exps_links'] = clean

        # 9) DEBUG: show cleaned link list
        if dbg:
            n_clean = len(clean) // 2
            print(f"[DBG] Clean exps_links pairs: {n_clean}, cleaned list (first 6 points) = {clean[:6]}")

        return memory_map_data
    
    def build_pcfg_from_memory(self):
        mg = self.models_manager.memory_graph
        emap = mg.experience_map
        current_id = mg.get_current_exp_id()
        exps_by_dist = mg.get_exps_organised(current_id)
        all_exps = emap.exps

        # --- Build graph from experience links ---
        graph = {exp.id: [link.target.id for link in exp.links] for exp in all_exps}
        print("[PCFG DEBUG] graph:", graph)
        print("[PCFG DEBUG] exps_by_dist =", exps_by_dist)
        print("[PCFG DEBUG] exps_by_dist type:", type(exps_by_dist))

        # --- Distance-based priors ---
        id_to_dist = {}
        total_weight = 0
        for exp in exps_by_dist:
            dist = (exp['x'] ** 2 + exp['y'] ** 2) ** 0.5
            id_to_dist[exp['id']] = dist
            total_weight += 1.0 / (dist + 1e-5)

        # --- Grammar rule collection ---
        rules = defaultdict(list)
        rules['EXPLORE'].append(('NAVPLAN', 1.0))

        for target_id, dist in id_to_dist.items():
            prob = (1.0 / (dist + 1e-5)) / total_weight
            rules['NAVPLAN'].append((f'GOTO_{target_id}', prob))
            rules[f'GOTO_{target_id}'].append((f'MOVESEQ_{current_id}_{target_id}', 1.0))

        # --- Collect known STEP transitions ---
        step_rules = set()
        for exp in all_exps:
            for link in exp.links:
                src = exp.id
                dst = link.target.id
                lhs = f'STEP_{src}_{dst}'
                rhs = f"'step({src},{dst})'"
                step_rules.add((lhs, rhs, 1.0))

        # --- Helper: Path-finding (BFS for simplicity) ---
        def find_paths(graph, start, goal, max_depth=5, max_paths=3):
            paths = []
            queue = deque([[start]])
            while queue and len(paths) < max_paths:
                path = queue.popleft()
                last = path[-1]
                if last == goal:
                    paths.append(path)
                elif len(path) < max_depth:
                    for neighbor in graph.get(last, []):
                        if neighbor not in path:
                            queue.append(path + [neighbor])
            return paths

        # --- Generate MOVESEQ rules via all valid path permutations ---
        for target_id in id_to_dist:
            moveseq_lhs = f"MOVESEQ_{current_id}_{target_id}"
            paths = find_paths(graph, current_id, target_id)
            if paths:
                for path in paths:
                    steps = [f"STEP_{a}_{b}" for a, b in zip(path, path[1:])]
                    rhs = " ".join(steps)
                    rules[moveseq_lhs].append((rhs, 1.0 / len(paths)))
            else:
                # Fallback to direct if no path (or self to self)
                rhs = f"STEP_{current_id}_{target_id}"
                rules[moveseq_lhs].append((rhs, 1.0))
                step_rules.add((rhs, f"'step({current_id},{target_id})'", 1.0))

        # --- Optional: hand-coded paths ---
        hardcoded_paths = {
            f'MOVESEQ_{current_id}_18': [f'STEP_{current_id}_19', 'STEP_19_18'],
            f'MOVESEQ_{current_id}_3':  [f'STEP_{current_id}_5', 'STEP_5_4', 'STEP_4_3'],
        }
        for lhs, steps in hardcoded_paths.items():
            if lhs not in rules:
                rhs = " ".join(steps)
                rules[lhs].append((rhs, 1.0))
                for step in steps:
                    s, d = step.replace("STEP_", "").replace("'", "").split("_")
                    step_rules.add((step, f"'step({s},{d})'", 1.0))

        # --- Final assembly of PCFG string ---
        pcfg_lines = []
        for lhs, productions in rules.items():
            total = sum(prob for _, prob in productions)
            for rhs, prob in productions:
                norm_prob = prob / total if total > 0 else 1.0
                pcfg_lines.append(f"{lhs} -> {rhs} [{norm_prob:.4f}]")

        for lhs, rhs, prob in step_rules:
            pcfg_lines.append(f"{lhs} -> {rhs} [{prob:.4f}]")

        print("[PCFG DEBUG] Final grammar string:\n" + "\n".join(pcfg_lines))
        grammar = PCFG.fromstring("\n".join(pcfg_lines))
        return grammar





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
  