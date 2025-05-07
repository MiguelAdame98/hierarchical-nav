# Update 2022
# =============================================================================
# Ghent University 
# IDLAB of IMEC
# Daria de Tinguy - daria.detinguy at ugent.be
# =============================================================================

# Original Source
# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
This is a full MemoryGraph implementation in python heavily based on the Ratslam implementation.
This implementation is based on Milford's original implementation [1]_ 
in matlab, and Christine Lee's python 
implementation [2]_. 

.. [2] https://github.com/coxlab/ratslam-python
.. [3] http://www.numpy.org/

'''
import torch
import dill
import cv2
import numpy as np
from .odometry import VisualOdometry,\
    PoseOdometry, ActionOdometry, Odometry, HotEncodedActionOdometry
from .view_cells import  TorchedViewCells, TorchedViewCell
from .pose_cells import PoseCells
from .experience_map import Experience, ExperienceMap
from .modules import *
# from matplotlib import pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3


class MemoryGraph(object):
    '''MemoryGraph implementation.

    The memory_graph is divided into 4 modules: visual odometry, view cells, pose
    cells, and experience map. This class also store the odometry and pose
    cells activation in order to plot them.
    '''

    def __init__(self, observation="camera", odom="action", 
                 **kwargs):
        '''Initializes the memory_graph modules.
        :param key: which key to use as observation
        :param odom: which key to use for odometry estimation
        '''
        self.observation_key = observation
        self.device = torch.device('cpu')

        self.local_position = kwargs.get('local_position', None)
        self.ghost_node_process = kwargs.get('ghost_node', False)
        self.odometry_key = odom
       
        if odom == "odom":
            self.odom = Odometry()
        elif odom == "action":
            self.odom = ActionOdometry()
        elif odom == "HEaction":
            self.odom = HotEncodedActionOdometry()
        elif odom == "pose":
            self.odom = PoseOdometry()
        else:
            self.odom = VisualOdometry(observation)

        self.view_cells = TorchedViewCells(observation, **kwargs)

        self.pose_cells = PoseCells(**kwargs)
        self.experience_map = ExperienceMap(**kwargs)
        
        Experience._ID = 0
        TorchedViewCell._ID = 0
        self.observation = None
        self.odometry = [[], [], []]
        self.pc = [[], [], []]

        

        #TODO: TEMPO fixed odom (no pose_cell in imp1):
        #self.odometry = [0.,0.,np.pi/2]
        
    def digest(self, observations, dt=0.1, adjust_map=True):
        # transform from torch format to numpy format
        # after migration this should no longer be required
        if "TensorDict" in str(type(observations)):
            observations = {key: val.numpy()
                            for key, val in observations.items()}
            if 'camera' in observations:
                observations["camera"] = np.transpose(
                observations["camera"], (1, 2, 0))
            if 'image' in observations:
                observations["image"] = np.transpose(
                observations["image"], (1, 2, 0))

        '''Execute a step of memory_graph algorithm for observations
        at the current time step.

        :param observations: a key-value dict of observations
        '''

        
                                                  
        self.observation = observations[self.observation_key]
        print("wtf is happening here",self.observation,observations[self.odometry_key])
        action_ob = observations[self.odometry_key]
        
        kwargs = {}
        kwargs['KL'] = observations.get('KL',None)
        if self.local_position:
            local_pose = observations.get(self.local_position,None)
            kwargs['local_position'] = local_pose

        

        # t1 = time.time()
        # get active pose and view cells
        # x_pc, y_pc, th_pc = self.pose_cells.active
        #TODO: clean later odom param (here we consider global odom directly from odom)
        x_pc, y_pc, th_pc = self.pose_cells.active
        print(x_pc, y_pc, th_pc)
        
        #if no observation, we consider previous one
        # if self.observation is None:
        #     self.observation = self.view_cells.prev_cell.template
        # get odom estimations
        vtrans, vrot = self.odom(action_ob, dt)
        x, y, th = self.odom.odometry
        print("odom", x, y, th)

        #TODO: TEMPO SIMPLIFICQTION
        #We want to create a view_cell ONLY if distance is far enough from prev one.
        #_,accum_delta_x, accum_delta_y  = self.experience_map.accumulated_delta_location(vtrans,vrot)
        #delta_exp = self.experience_map.get_delta_exp(0,0,accum_delta_x, accum_delta_y)
        print(f"[DEBUG] odometry → vtrans={vtrans:.6f}, vrot={vrot:.6f}, odom pose=({x:.2f}, {y:.2f}, {th:.2f})")

        
        rotation_only = abs(vtrans) < 1e-4 and abs(vrot) > 0
        print(f"[DEBUG] rotation_only? {rotation_only}")

        if rotation_only and self.experience_map.current_exp is not None:
            # override: don’t create a new cell if we only turned
            print(f"[DEBUG] Pure rotation detected. Forcing no new experience creation.")
            delta_pc_above_thresold = False
            kwargs['current_exp_id']= self.experience_map.current_exp.id
            kwargs['delta_exp_above_thresold'] = False
        else:
            if self.experience_map.current_exp is not None:
                x_exp = self.experience_map.current_exp.x_pc
                y_exp = self.experience_map.current_exp.y_pc
                print(f"[DEBUG] Translational step. last exp at (x_pc={x_exp}, y_pc={y_exp}), now at (x_pc={x_pc}, y_pc={y_pc})")
                delta_pc = self.experience_map.get_delta_exp(x_exp, y_exp, x_pc, y_pc)
                delta_pc_above_thresold = self.experience_map.delta_pc_above_thresold(delta_pc)
                print(f"[DEBUG] computed delta_pc={delta_pc:.4f}, above_threshold={delta_pc_above_thresold}")
                kwargs['current_exp_id'] = self.experience_map.current_exp.id
            else:
                delta_pc_above_thresold = 100
                kwargs['current_exp_id'] = None
        print(delta_pc_above_thresold)
        kwargs['delta_exp_above_thresold'] = delta_pc_above_thresold
        
        view_cell, view_cell_copy = self.view_cells(self.observation, x_pc, y_pc, th_pc, **kwargs)
        #view_cell_copy: same as view cell but with a different id
       
        print(self.view_cells.cells,len(self.view_cells.cells))
        if view_cell is None:
            #this only happens if we start memory_graph with a memory 
            # and have yet to define the place (no observation yet)
            return

        # update pose cells
        print("this goes into a pose cell",self.observation, view_cell,vtrans,vrot,adjust_map,"local pose",local_pose)
        x_pc, y_pc, th_pc = self.pose_cells(view_cell, vtrans, vrot)
        print("this goes into a pose cell",x_pc, y_pc, th_pc)           
        if self.experience_map.current_exp is None:
            # reset the pose cells to this activity
            print("reset the pose cell")
            self.pose_cells.reset(view_cell.x_pc,
                                  view_cell.y_pc,
                                  view_cell.th_pc)
            

       
        print("this are the view_cell pre exp:",view_cell.exps,view_cell,view_cell.id)
        # update experience map
        self.experience_map(view_cell, vtrans, vrot,
                            x_pc, y_pc, th_pc, adjust_map, local_pose, view_cell_copy)
        if self.experience_map.current_exp is not None:
            self.view_cells.update_prev_cell(self.experience_map.current_exp.view_cell)
        #TEST
        if self.experience_map.current_exp is not None:
            pose = observations.get(self.local_position,None)
            print('memory_graph pose', pose)
            try:
                pose_GP_facing = local_encoded_orientation_to_global_facing(pose[2], self.experience_map.current_exp.init_local_position[2], self.experience_map.current_exp.facing_rad)
                test = encoded_orientation_given_facing(pose_GP_facing, self.experience_map.current_exp.init_local_position[2], self.experience_map.current_exp.facing_rad)
                print('must be True', test == pose[2], test, pose[2])
            except TypeError:
                 print('type error')

        # for tracking and plotting
        self.odometry[0].append(x)
        self.odometry[1].append(y)
        self.odometry[2].append(th)
        self.pc[0].append(x_pc)
        self.pc[1].append(y_pc)
        self.pc[2].append(th_pc)

#============= Ghost node creation ================#
    def create_ghost_exps(self, exp_id:int = None, steps_margin:int = 3)-> None:
        ''' Params:
        exp_id: the experience id we want to create ghost nodes for
        steps_margin: how further from the limits of prev place do we want to create the ghost node at
        Ghost nodes are created at the GP of this exp door poses + some margin'''
        
        if not self.ghost_node_process :
            return
        
        if exp_id is None:
            ref_exp = self.current_exp
        else:
            ref_exp = self.get_exp(exp_id)
        #No exp? then don't do anything
        if ref_exp is None:
            return
        #TODO: this is specific to ghost created AT ALL relevant poses, 
        # might not be pertinent in a non minigrid env
        exp_local_pose = ref_exp.init_local_position
        exp_GP = [ref_exp.x_m, ref_exp.y_m, ref_exp.facing_rad]

        relevant_poses = self.get_exp_relevant_poses(ref_exp)
        GP_relevant_poses = []
        for pose in relevant_poses:
            GP_pose = convert_LP_to_GP(exp_GP, exp_local_pose, pose)
            for margin in range(self.get_delta_exp_threshold() + 1):
                GP_pose = self.odom.position_applying_motion(GP_pose, [1,0,0])
            GP_relevant_poses.append(GP_pose)
        self.experience_map.create_ghost_exps(ref_exp, GP_relevant_poses)
          
#============== GET METHODS ================#
    def get_delta_exp_threshold(self):
        """ min distance between two exps"""
        return self.experience_map.DELTA_EXP_THRESHOLD
    
    def get_exp_relevant_poses(self, exp_id= None):
        ''' Extract  exp view cell relevant_poses'''
        if exp_id == None:
            exp = self.experience_map.current_exp
        else:
            exp = self.experience_map.get_exp(exp_id)
        if exp == None:
            return []
        print("relevant poses", len(exp.view_cell.relevant_poses.copy()),exp.view_cell.relevant_poses.copy())
        return exp.view_cell.relevant_poses.copy()

    def get_next_place_view_id(self, door_pose):
        ''' old strategy to link exp through doors, but it is dependant on correct imagination (door imagined + ~correct position)'''
        GP_door_pose = [0,0,0]
        current_exp = self.experience_map.current_exp
        global_position = [current_exp.x_m, current_exp.y_m, current_exp.facing_rad]
        exp_local_pose = current_exp.init_local_position

        door_dist_to_start_pose = [int(door_pose[0] - exp_local_pose[0]), int(door_pose[1] - exp_local_pose[1])]
        #GP localisation of the given door in current exp
        entry_door_facing_rad = local_encoded_orientation_to_global_facing(0, exp_local_pose[2], global_position[2])
        
        print('current exp Lp and GP',exp_local_pose, global_position)
        GP_door_pose[0] = global_position[0] +  (door_dist_to_start_pose[0] * np.cos(entry_door_facing_rad) - door_dist_to_start_pose[1] * np.sin(entry_door_facing_rad))
        GP_door_pose[1] = global_position[1] +  (door_dist_to_start_pose[0] * np.sin(entry_door_facing_rad) + door_dist_to_start_pose[1] * np.cos(entry_door_facing_rad))
        
        #get the GP orientation of the door when facing the room (given the view when facing it)
        door_room_facing = (door_pose[2]+2) %4
        door_facing_room_rad = local_encoded_orientation_to_global_facing(door_room_facing, exp_local_pose[2], global_position[2])
        
        print('CURRENT PLACE Door ', door_pose,' global pose', GP_door_pose[:2], 'orientation facing room', door_facing_room_rad,' door lp dist to init pose', door_dist_to_start_pose, )
        print('exp xpc, ypc tph + init lp', [current_exp.x_pc, current_exp.y_pc, current_exp.th_pc], current_exp.init_local_position)

        for link in current_exp.links:
            print()
            linked_exp = link.target
            if linked_exp.ghost_exp == True:
                print('IMPLEMENT WHAT HAPPENS IF THERE IS A GHOST NODE')
                continue
            
            linked_exp_global_position =   [linked_exp.x_m, linked_exp.y_m, linked_exp.facing_rad] 
            linked_exp_view = self.experience_map.get_exp_view_cell_content(linked_exp.id)
            
            linked_exp_local_pose = linked_exp.init_local_position
            linked_exp_door_poses = linked_exp_view.relevant_poses

            linked_exp_entry_door_facing_rad = local_encoded_orientation_to_global_facing(0, linked_exp_local_pose[2], linked_exp_global_position[2])
            print('exp',linked_exp.id ,' Lp and GP',linked_exp_local_pose, linked_exp_global_position)
            print('exp xpc, ypc, tph + init lp', [linked_exp.x_pc, linked_exp.y_pc, linked_exp.th_pc], linked_exp.init_local_position)
            #get the GP orientation of the entry doors of the linked exp
            for link_exp_door_pose in linked_exp_door_poses:
                linked_exp_GP_door_pose = [0,0,0]
               
                door_dist_to_start_pose = [int(link_exp_door_pose[0] - linked_exp_local_pose[0]), int(link_exp_door_pose[1] - linked_exp_local_pose[1])]
                linked_exp_GP_door_pose[0] = linked_exp_global_position[0] +  (door_dist_to_start_pose[0] * np.cos(linked_exp_entry_door_facing_rad) - door_dist_to_start_pose[1] * np.sin(linked_exp_entry_door_facing_rad))
                linked_exp_GP_door_pose[1] = linked_exp_global_position[1] +  (door_dist_to_start_pose[0] * np.sin(linked_exp_entry_door_facing_rad) + door_dist_to_start_pose[1] * np.cos(linked_exp_entry_door_facing_rad))
                
                linked_exp_GP_door_pose[2] = local_encoded_orientation_to_global_facing(link_exp_door_pose[2], linked_exp_local_pose[2], linked_exp_global_position[2])
                print('linked exp ', linked_exp.id,'considered door ',link_exp_door_pose,' GP door', linked_exp_GP_door_pose, 'dist to this exp init lp', door_dist_to_start_pose)
                
                delta_gp = [np.abs(linked_exp_GP_door_pose[0] - GP_door_pose[0]), np.abs(linked_exp_GP_door_pose[1] - GP_door_pose[1])]
                print('delta GP in x, y between this door and current exp door', delta_gp)
                #If inversed orientation match the current door entry, then it means we are facing the door leading to this exp
                if np.abs(linked_exp_GP_door_pose[2] - door_facing_room_rad) < np.pi/3 and delta_gp[0] < 7 and delta_gp[1] < 7:
                    relevant_poses = self.get_exp_relevant_poses(linked_exp.id)
                    door_pose_from_new_place = [pose for pose in relevant_poses if pose[2] == link_exp_door_pose[2]][0]

                    print('returning linked exp ', linked_exp.id, ' view id ',linked_exp_view.id, 'door pose from this view perspective', door_pose_from_new_place)
                    return  linked_exp.id, door_pose_from_new_place

        return -1, []
    
    def get_exps(self, wt_links:bool = False)->dict:
        ''' 
        give back all the experiences as dict organised in a list with id, map position, 
        view cell template and infos (if any). They will be organised by exp id
        '''

        return self.experience_map.get_exps(wt_links = wt_links)

    def get_exps_organised(self, exp_id:int = None, from_current_pose:bool=False) -> dict:
        '''
        organise the exps from closest to further from given exp, in term of distance
        '''
        
        exps = self.get_exps()
        if from_current_pose:
            #current Global Position
            agent_gp = self.get_global_position()
            goal_exp = {'id':-1, 'x':agent_gp[0], 'y':agent_gp[1], 'facing':agent_gp[2] }
        elif exp_id is None:
            #current exp
            goal_exp = self.experience_map.get_exp_as_dict()
        elif isinstance(exp_id, int) and exp_id >= 0:
            #given exp
            goal_exp = exps[exp_id]
            
            #print('goal exp',goal_exp['id'])
        if not goal_exp :
            return exps
        exps.sort(key=lambda i: sort_by_distance(i, goal_exp))
        return exps
        
    def get_current_exp_and_view_id(self) -> tuple[int,int]:
        exp_id = self.experience_map.get_current_exp_id()
        view_cell_id = self.get_exp_view_cell_id(exp_id)

        return exp_id, view_cell_id

    def get_current_exp_id(self) ->int:
        exp_id = self.experience_map.get_current_exp_id()
        return exp_id
        
    def get_exp_view_cell_id(self, id:int)-> int:
        ''' given location id, get view cell id'''
        view_cell = self.get_exp_view_cell(id)
        if view_cell is not None:
            return view_cell.id
        else:
            return -1
    
    def get_view_cell_template(self, id:int)-> torch.Tensor: 
        return self.view_cells.templates[id]
    
    def get_exp_view_cell(self, id:int) -> object:   
        ''' given location id, extract view id '''
        return self.experience_map.get_exp_view_cell_content(id)
    
    def get_exp_place(self, id:int) -> np.ndarray:
        ''' given exp id return the observation.
        If exp_id = -1 return current exp observation 
        if possible (else observation is None)'''
        if id < 0:
            if self.experience_map.current_exp is not None:
                id = self.experience_map.current_exp.id
            else:
                id = -1
    
        view_cell = self.get_exp_view_cell(id)
        observation = self.extract_observation(view_cell)
        return observation
    
    def get_global_position(self) -> list:
        """ get agent current global position"""
        return self.experience_map.get_global_position()

    def get_exp_global_position(self, exp:object=-1):
        return self.experience_map.get_exp_global_position(exp)
#============== ? ================#
    def memorise_poses(self,poses:list, exp_id:int =-1):
        ''' Given exp id and list of door poses, save them in exp view cell'''
        
        if exp_id < 0:
            exp = self.experience_map.current_exp
        else:
            exp = self.experience_map.get_exp(exp_id)
        if exp is None:
            return
        exp.view_cell.relevant_poses = [list(pose) for pose in poses]
        print("poses_memorized",exp.view_cell.relevant_poses)

    def extract_observation(self, view_cell:object) -> np.ndarray:
        if view_cell is not None:
            return view_cell.template
        else:
            return None
      
    def current_distance_to_exp(self, exp_id:int=None) -> float:
        """ get global distance between two locations"""
        if exp_id == None:
            exp = self.experience_map.current_exp
        else:
            exp = self.experience_map.get_exp(exp_id)
        if exp == None:
            return []
        current_GP = self.get_global_position()
        exp_GP = [exp.x_m, exp.y_m, exp.facing_rad] 
        #print('give me the exp id, the exp GP and current GP', exp_id, exp_GP, current_GP)
        dist_exp = euclidian_distance(exp_GP, current_GP)

        return dist_exp
        
    def find_goal_in_memory(self):
        ''' We have a memory of a specific observation we want to go to, 
        we want to know where it corresponds to'''
        pass
        
    def find_door_linking_exps(self, goal_exp_id:int, start_exp_id:int = None)->list:
        '''
        Given goal exp and start exp (option) find the connecting door from start exp 
        '''
        if start_exp_id == None:
            start_exp = self.experience_map.current_exp
        else:
            start_exp = self.experience_map.get_exp(start_exp_id)

        start_exp_global_position = self.experience_map.get_exp_global_position(start_exp)
        start_exp_local_pose = start_exp.init_local_position
        start_exp_door_poses = self.get_exp_relevant_poses(start_exp.id)

        goal_exp = self.experience_map.get_exp(goal_exp_id)
        goal_exp_global_position = self.experience_map.get_exp_global_position(goal_exp)


        for door_pose in start_exp_door_poses:
            GP_door_pose = convert_LP_to_GP(start_exp_global_position, start_exp_local_pose, door_pose)
            angle_door_goal_exp = angle_between_two_poses(goal_exp_global_position, GP_door_pose)
             
            
            print('Between door', door_pose,' GP:', GP_door_pose , 'and exp', goal_exp.id, 'GP:', goal_exp_global_position, 'the angle is:', angle_door_goal_exp)
            print('the angular difference considering door orientation is:', np.rad2deg(np.round(clip_rad_180(GP_door_pose[2] - angle_door_goal_exp),2)))
            #The angle between the door and the linked exp can't be above 90 def considering door orientation. Else not correct direction
            rad_diff = clip_rad_180(GP_door_pose[2] - angle_door_goal_exp)
            if rad_distance_under_threshold(np.round(rad_diff,2), np.pi/2) :
                print('door',door_pose, ' links experience', start_exp.id, 'and', goal_exp.id )
                return door_pose
            
        return []

    def linked_exp_behind_door(self, door_pose:list):
        '''
        Confirm if door connects to another experience.
        The angular distance is considered (and not door matching) in case the memory 
        isn't correct and the connected exp is missing doors or they are not estimated at the correct place

        Thus this function check the angular distance one way, and just the angle matching between door pose the other way, 
        as missing connecting door pose is not really a problem
        
        '''
        #---- Get experience Reference frame and convert it to Global Frame ----#
        
        current_exp_id = self.get_current_exp_id()
        print("this is the current_exp_id",current_exp_id)
        if current_exp_id < 0:
            print("we returned nothing")
            return -1, []
        current_exp = self.experience_map.get_exp(current_exp_id)
        print("this is the current_exp", current_exp)
        exp_global_position = self.experience_map.get_exp_global_position(current_exp)
        exp_local_pose = current_exp.init_local_position
        

        GP_door_pose = convert_LP_to_GP(exp_global_position, exp_local_pose, door_pose)
        
        print('current exp Lp and GP',exp_local_pose, exp_global_position)
        print('Considering door pose', door_pose, GP_door_pose)
        
        
        exp_options = []
        #Check if exp door goes toward connected exp
        for link in current_exp.links:
            print("a",link)
            linked_exp = link.target
            if linked_exp.ghost_exp == True:
                print('IMPLEMENT WHAT HAPPENS IF THERE IS A GHOST NODE')
                continue

            linked_exp_global_position = self.experience_map.get_exp_global_position(linked_exp)
            print("this is the global position", linked_exp_global_position)
            '''
            We consider that a door leading to an exp cannot have an angular displacement between the 2
            of more than 90'.
            If they do have such a displacement, the door likely doesn't lead toward this exp.
            '''
            #the angular distance between the two poses. 
            angle_between_door_exp = angle_between_two_poses(linked_exp_global_position, GP_door_pose)
            print("angle_between_door",angle_between_door_exp)
            rad_diff = clip_rad_180(GP_door_pose[2] - angle_between_door_exp)
            print("rad_diff",rad_diff)
            print('Between door', door_pose,' GP:', GP_door_pose , 'and exp', linked_exp.id, 'GP:', linked_exp_global_position, 'the angular distance is:', angle_between_door_exp)
            print('the angular difference considering door orientation is:', np.rad2deg(rad_diff))
            #The angle between the door and the linked exp can't be above 90 def considering door orientation. Else not correct direction
            if rad_distance_under_threshold(rad_diff, threshold=np.pi/2) :
                print('linked exp', linked_exp.id, 'is likely connected through door pose', door_pose)

                #then we search which door connect to current exp from the next room 
                #NOTE: the implementation was chosen in 2 steps like this to paliate model imagination errors (imagined room doesn't have doors on this wall for instance)
                                
                linked_exp_door_poses = self.get_exp_relevant_poses(linked_exp.id)
                linked_exp_local_pose = linked_exp.init_local_position
                linked_exp_connecting_door = self.find_connecting_door_to_given_door(GP_door_pose, linked_exp_door_poses,\
                                                                                    linked_exp_local_pose,linked_exp_global_position)
                                
                print('door pose', door_pose, 'connected to', linked_exp.id ,' via door pose', linked_exp_connecting_door)
                dist_exp = euclidian_distance(linked_exp_global_position, exp_global_position)
                exp_options.append([linked_exp.id, linked_exp_connecting_door, dist_exp])
        
        #We consider the further connected exp in that direction to avoid exp too close (likely same room)
        if len(exp_options) > 0 :
            linked_exp = max(exp_options, key=lambda x: x[-1])[:2]
            return linked_exp[0], linked_exp[1]
        
        else:
            print('no known exp behind', door_pose)
            return -1, []
        
    def find_connecting_door_to_given_door(self,GP_destination_pose:list, linked_exp_door_poses:list,\
                                            linked_exp_local_pose:list, linked_exp_global_position:list):
        """
        We want to find a connection to the given door from another experience 
        GP_destination_pose: objective door
        linked_exp_door_poses: list of options to connect doors
        linked_exp_local_pose: exp local position
        linked_exp_global_position: exp global position
        """
        linked_exp_connecting_door = []
        for link_exp_door_pose in linked_exp_door_poses:
            linked_exp_door_inverse_facing = local_encoded_orientation_to_global_facing((link_exp_door_pose[2]+2) %4, linked_exp_local_pose[2], linked_exp_global_position[2])
            if rad_distance_under_threshold(linked_exp_door_inverse_facing - GP_destination_pose[2], threshold=np.pi/3):
                #print('connecting doors', link_exp_door_pose, 'and', door_pose)
                linked_exp_connecting_door = link_exp_door_pose
                break
        return linked_exp_connecting_door
    
    def convert_pose_orientation_from_start_ref_frame_to_another(self,pose:list,  start_exp_id:int= None, goal_exp_id:int=None)-> int:
        '''
        given a pose and the starting and goal experience we want to use the reference frame of
        convert pose orientation from a place reference frame to another place reference frame
        return the orientation in the goal ref frame
        '''
        if goal_exp_id == None:
            g_exp = self.experience_map.current_exp
        else:
            g_exp = self.experience_map.get_exp(goal_exp_id)

        if start_exp_id == None:
            s_exp = self.experience_map.current_exp
        else:
            s_exp = self.experience_map.get_exp(start_exp_id)
        
        #we get globalframe orientation of a pose using starting exp local reference frame
        pose_GP_facing = local_encoded_orientation_to_global_facing(pose[2], s_exp.init_local_position[2], s_exp.facing_rad)
        #we obtain the corresponding local orientation in the goal exp local reference frame
        pose_place_orientation = encoded_orientation_given_facing(pose_GP_facing, g_exp.init_local_position[2], g_exp.facing_rad)
        # pose_place_oriented = pose
        # pose_place_oriented[2]= pose_place_orientation
        return pose_place_orientation

#============== CLASS CREATION METHODS ================#
    def idify(self):
        # break circular dependencies by storing ids instead
        for vc in self.view_cells.cells:
            exp_ids = []
            for exp in vc.exps:
                exp_ids.append(exp.id)
            vc.exps = exp_ids

        for exp in self.experience_map.exps:
            exp.view_cell = exp.view_cell.id
            for link in exp.links:
                link.parent = link.parent.id
                link.target = link.target.id

    def objectify(self):
        # reestablish the object links
        for vc in self.view_cells.cells:
            exps = []
            for exp_id in vc.exps:
                exps.append(self.experience_map.exps[exp_id])
            vc.exps = exps

        for exp in self.experience_map.exps:
            exp.view_cell = self.view_cells.cells[exp.view_cell]
            for link in exp.links:
                link.parent = self.experience_map.exps[link.parent]
                link.target = self.experience_map.exps[link.target]

    def save(self, file:str):
        f = open(file, 'wb')
        self.idify()
        d = {
            "view_cells": self.view_cells.cells,
            "experience_map": self.experience_map.exps,
        }
        dill.dump(d, f)
        f.close()
        self.objectify()

    def load(self, file:str):
        f = open(file, 'rb')
        d = dill.load(f)
        self.view_cells.cells = d["view_cells"]
        self.view_cells.load_memorised_templates()

        self.experience_map.exps = d["experience_map"]
        self.experience_map.size = len(self.experience_map.exps)
        Experience._ID = self.experience_map.size
        TorchedViewCell._ID = len(self.view_cells.cells)

        self.observation = None
        self.odometry = [[], [], []]
        self.pc = [[], [], []]
        f.close()

        self.objectify()
