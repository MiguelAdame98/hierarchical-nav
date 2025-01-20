import env_specifics.minigrid_maze_wt_aisles_doors.minigrid_maze_modules as minigrid_maze

def call_env_place_range(env:str, reduction:int=0) -> list :
    ''' Set the limits [x,y] of the place range (for hypothesis range)
    Minigrid_maze: considers the rooms number of tiles only
    '''
    if 'Minigrid_maze_aisle_wt_doors' in env:
        pose_options = minigrid_maze.minigrid_maze_aisles_doors_pose_range(env, reduction=reduction)
        return pose_options
    else:
        raise('call_env_place_range, '+ env + ' not implemented')

def call_env_entry_poses_assessment(env:str, entry_poses:list) -> list :
    ''' Determine which poses could possibly lead toward the place
    Minigrid_maze: considers the rooms number of tiles only
    '''
    if 'Minigrid_maze_aisle_wt_doors' in env:
        entry_poses = minigrid_maze.poses_leading_in_room(entry_poses)
        return entry_poses
    else:
        raise('call_env_place_range, '+ env + ' not implemented')

def call_env_remove_double_poses(env:str, poses:list) -> list:
    ''' 
    remove unecessary poses
    Minigrid_maze: consider only 1 pose by orientation, the 
    '''
    if 'Minigrid_maze_aisle_wt_doors' in env:
        door_poses = minigrid_maze.remove_double_orientation(poses)
        return door_poses
    else:
        raise('call_env_place_range, '+ env + ' not implemented')
    
def call_env_number_of_entry_points(env:str) -> int:
    ''' 
    how many entry points are expected by env
    '''
    if 'Minigrid_maze_aisle_wt_doors' in env:
        num_entry_points = 4
        return num_entry_points
    else:
        raise('call_env_place_range, '+ env + ' not implemented')
    
def call_process_limit_info(env:str, pose:list) -> dict:
    '''
    initiating a dictionnary containing the relative place position, place info,the direction 
    (axe + direction +/-)necessary to reach the pose, the next place info    
    '''
    if 'Minigrid_maze_aisle_wt_doors' in env:
        door_info = minigrid_maze.process_limit_info(pose)
        return door_info
    else:
        raise('call_env_place_range, '+ env + ' not implemented')
    
def call_from_door_view_to_door_pose(env:str,pose:list) -> list:
    '''
    ONLY MINIGRID_MAZE_AISLE_WT_DOOR
    When we have a pose associated to a door, 
    it is the pose of the view of the door, not the door pose
    '''
    if 'Minigrid_maze_aisle_wt_doors' in env:
        door_pose = minigrid_maze.from_door_view_to_door_pose(pose)
        return door_pose
    else:
        raise('call_from_door_view_to_door_pose is specific to Minigrid_maze_aisle_wt_doors, please refactor')
    
def call_get_place_behind_limit(env:str, manager:object, pose:list):
    """   get the eperience behind the given limit of current place"""
    if 'Minigrid_maze_aisle_wt_doors' in env:
        expected_exp_id, door_pose_from_new_place = minigrid_maze.get_place_behind_door(manager, pose)
        return expected_exp_id, door_pose_from_new_place
    else:
        raise('call_from_door_view_to_door_pose is specific to Minigrid_maze_aisle_wt_doors, please refactor')
    
def find_preferred_features_in_img(env:str, ob, agent_pose:list, goal_features:list) -> tuple[list, float]:
    """ search for goal feature matching in given ob as a Tensor, 
    return the pose of the goal in observation and how likely it is to be correct"""
    if 'Minigrid_maze_aisle_wt_doors' in env:
        goal_pose, colour_likelihood = minigrid_maze.find_tile_colour_in_img(ob, agent_pose, goal_features)
        return goal_pose, colour_likelihood
    else:
        raise('call_from_door_view_to_door_pose is specific to Minigrid_maze_aisle_wt_doors, please refactor')