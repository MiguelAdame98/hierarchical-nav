from itertools import product
from control_eval.input_output import load_h5
import numpy as np
import torch
from navigation_model.Services.model_modules import torch_and_sample_observations
from navigation_model.Processes.AIF_modules import mse_elements
from navigation_model.Processes.motion_path_modules import action_to_pose
from navigation_model.visualisation_tools import convert_tensor_image_to_array


#TODO: add this call in env setup since it's particular to minigrid env      
def set_door_view_observation(manager: object)-> None:
    ''' Give the static info of what a door looks like to the model '''
    #Static Memory of a door view
    door_view_file = 'env_specifics/minigrid_maze_wt_aisles_doors/door_view.h5'
    door_view = load_h5(door_view_file)
    door_view = torch_and_sample_observations(door_view, manager.get_observations_keys(), manager.get_manager_sampling())
   
    manager.set_env_relevant_ob(door_view)
    
#NOTE: sensitivity 0.18 for minigrid_aisle_wt_doors and current allocentric model
def is_agent_at_door(manager:object, sensitivity:float= 1) -> bool:
    '''
    verify if the predicted observation matches a door view. if so, save in short-term memory 
    '''
    best_place_hypothesis = manager.get_best_place_hypothesis()
    if 'image_predicted' in best_place_hypothesis and best_place_hypothesis['image_predicted'] is not None:
        door_image = manager.get_env_relevant_ob()['image']
        predicted_image = best_place_hypothesis['image_predicted'].squeeze(1)
        if predicted_image.shape[0] > door_image.shape[0]:
            predicted_image = predicted_image[0]
        mse_door = mse_elements(predicted_image, door_image)
        if manager.mse_under_threshold(mse_door, sensitivity):
            pose = list(best_place_hypothesis['pose'])
            manager.save_pose_in_memory(pose)
            return True
    return False

def is_agent_at_door_given_ob(manager:object, pred_image, pose:list, sensitivity:float= 1) -> bool:
    '''
    verify if the predicted observation matches a door view. if so, save in short-term memory 
    '''
    door_image = manager.get_env_relevant_ob()['image']
    pred_image = pred_image.squeeze(1)
    if pred_image.shape[0] > door_image.shape[0]:
            pred_image = pred_image[0]
    mse_door = mse_elements(pred_image, door_image)
    if manager.mse_under_threshold(mse_door, sensitivity):
        pose = list(pose)
        manager.save_pose_in_memory(pose)
        return True
    return False



def minigrid_maze_aisles_doors_pose_range(env:str, reduction:int = 0)-> list :
        '''
        determine the squared range limits of the rooms deepending on its size
        '''
        #Poses options in room (+ a bit of aisle)
        if '4' in env:
            x_range = [0,8]
            y_range = [-3,3]
        elif '5' in env:
            x_range = [0,9]
            y_range = [-4,4]
        elif '6' in env:
            x_range = [0,10]
            y_range = [-5,5]
        elif '7' in env:
            x_range = [0,11]
            y_range = [-6,6]
        elif '8' in env:
            x_range = [0,12]
            y_range = [-7,7]
        else:
            raise(env +'length not accounted for place range')
        return pose_option_setup(x_range, y_range, reduction)

def pose_option_setup(x_range:list, y_range:list, reduction:int) -> list:
        '''  form a list of all possible poses in the range, 
        the reduction is used to restrain those options'''
        x = list(range(x_range[0]+reduction, x_range[1]+1-reduction)) 
        y = list(range(y_range[0]+reduction, y_range[1]+1-reduction)) 
        theta = list(range(0, 4))

        pose_options = list(product(*[x,y,theta]))
        return list(map(list, pose_options)) #all pose options as a list of list

def poses_leading_in_room(door_poses:list) -> list:
    """ in this env the entry point are the aisles, 
    thus we add positions in aisle around the door pose given
    """
    pose_options = []
    for p in door_poses:
        p = np.array(p)
        p[2] = (p[2]+2) % 4 #we want to face the room, not the door
        pose_options.append(list(p))
        pose_options.append(list(action_to_pose([1,0,0], p)))
        back_pose = action_to_pose([-1,0,0], p)
        pose_options.append(list(back_pose))
        pose_options.append(list(action_to_pose([-1,0,0], back_pose)))
    
    return pose_options

def remove_double_orientation(poses:list) -> list:
    '''
    we expect [[mse,p]...] as input sorted from best to worst
    erase all the duplicate poses with same orientation,
    keeping only the best one.
    '''   
    #TODO: DO A CHECK OF SHAPE TO AVOID ERROR

    for o in range(4):
        idx = [i for i, x in enumerate(poses[:,1]) if x[2] == o][::-1]
        if len(idx) > 1:
            poses = np.delete(poses, idx[:-1], axis=0)

    return poses
            

def process_limit_info(pose:list) -> dict:
    ''' we create a dictionnary containing the door pose info
    This i
    '''
    pose = np.array(pose)
    forward_pose_front_door = action_to_pose([1,0,0], pose)
    motion = forward_pose_front_door - pose 
    motion_index = np.nonzero(motion)

    door_info = {}
    door_info['door_pose'] = pose
    door_info['motion_axis'] = motion_index[0]
    door_info['direction'] = motion[motion_index[0]]
    door_info['connected_place'] = None
    door_info['origin_place'] = None
    door_info['exp_connected_place'] = None
    door_info['connected_place_door_pose'] = []
    return door_info

def from_door_view_to_door_pose(pose:list) -> list:
    for i in range(2):
        pose = action_to_pose([1,0,0], np.array(pose))
    return pose

def get_place_behind_door(manager:object, pose:list) ->tuple[int, list] :
        if pose is not None:
            expected_exp_id, door_pose_from_new_place = manager.memory_graph.linked_exp_behind_door(pose)
        else:
            expected_exp_id, door_pose_from_new_place = -1, []
        return expected_exp_id, door_pose_from_new_place 


def find_tile_colour_in_img(observation:torch.Tensor, agent_pose:list, colour_range:np.ndarray)-> tuple[list, float]:
    """ search for a tile having a colour in colour_range range
    extract this tile pose given the agent pose corresponding to ob
    This assumes an RGB observation of 56x56 representing 7 tiles and the agent pose is at the bottom middle
    return the pose of the tile in observation and how likely it is to be correct 
    (how much in the middle of range)
    """

    #Static with our Maze minigrid environment 
    # and observation as 56x56 with 7 tile observed and agent in bottom middle position
    tile_size = 8
    row_col_tile_nb = 7
    img_agent_pose = np.array([6,3]) #x,y in img ref frame

    if isinstance(agent_pose, torch.Tensor):
        if len(agent_pose.shape) > 1:
            agent_pose = torch.mean(agent_pose, dim=list(range(len(agent_pose.shape)-1)))
        agent_pose = agent_pose.cpu().detach().numpy()
        #Make sure to have only 1 image
    if len(observation.shape) > 3:
        observation= torch.mean(observation, dim=list(range(len(observation.shape)-3)))
    img = convert_tensor_image_to_array(observation)

    per_colour_likelihood = 0
    diff_img_pose = None
    tile_averaged_img = np.zeros(img.shape)
    for row_tile in range(row_col_tile_nb):
        for col_tile in range(row_col_tile_nb):
            tile_img = img[row_tile*tile_size: (row_tile+1)*tile_size,
                            col_tile*tile_size: (col_tile+1)*tile_size]
            tile_RGB_average = np.average(tile_img, axis = (0,1))
            # print('tile extracted', [row_tile*tile_size, (row_tile+1)*tile_size,
            # col_tile*tile_size, (col_tile+1)*tile_size])
            # print('tile RGB average',tile_RGB_average, type(tile_RGB_average), tile_RGB_average/255)
            tile_averaged_img[row_tile*tile_size: (row_tile+1)*tile_size,
                        col_tile*tile_size: (col_tile+1)*tile_size]= tuple(tile_RGB_average/255)
            if np.all(colour_range[0] <= tile_RGB_average) and np.all(tile_RGB_average <= colour_range[1]):
                #if we are above the colour range limit we save this pose
                colour_img_pose = np.array([row_tile, col_tile])
                diff_img_pose = abs(img_agent_pose - colour_img_pose)
                per_colour_likelihood = colour_likelihood(tile_RGB_average, colour_range)
                if colour_img_pose[1] < img_agent_pose[1]:
                    diff_img_pose[1] = colour_img_pose[1] - img_agent_pose[1]
                #print(' col and row contains WHITE TILE', col_tile, row_tile)
    wt_pose= None
    if diff_img_pose is not None:
        x = agent_pose[0] + (diff_img_pose[0] * np.cos(agent_pose[2]*np.pi/2) - diff_img_pose[1] * np.sin(agent_pose[2]*np.pi/2))
        y = agent_pose[1] + (diff_img_pose[0] * np.sin(agent_pose[2]*np.pi/2) + diff_img_pose[1] * np.cos(agent_pose[2]*np.pi/2))
        wt_pose = [round(x),round(y), int(agent_pose[2])]
        # plt.figure()
        # plt.title('agent p:' + str(agent_pose)+ ', wt pose:' +str(wt_pose))
        # plt.imshow(tile_averaged_img)
        # plt.show()
    return wt_pose, per_colour_likelihood

def RGB_euclidean_distance(rgb1:list, rgb2:list) -> float:
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    return np.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)

def colour_likelihood(rgb_colour:np.ndarray, colour_range:list) -> int:
    ''' 
    Define how well centered the value is 
    between the two ranges in percentage
    In the centre: 100%, on a limit colour range: 0%
    '''
    center_1 = (colour_range[0] + colour_range[1]) / 2
    max_colour_dist = RGB_euclidean_distance(center_1, colour_range[1])

    dist_to_center = np.linalg.norm(rgb_colour - center_1)
    per_dist_to_center = round(100- (dist_to_center/ max_colour_dist) * 100)
    return per_dist_to_center