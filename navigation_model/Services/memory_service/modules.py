import numpy as np

def euclidian_distance(d1:list, d2:list)->int:
    delta_exp = np.sqrt(
        min_delta(d1[0],d2[0], np.inf)**2 +
        min_delta(d1[1],d2[1], np.inf)**2 
        #+ min_delta(self.current_exp.th_pc, th_pc, self.DIM_TH)**2
        )
    return delta_exp
    # vector_exps = np.subtract(d1, d2)
    # return np.sqrt(vector_exps[0]**2 + vector_exps[1]**2)
    
def min_delta(d1, d2, max_):
    delta = np.min([np.abs(d1 - d2), max_ - np.abs(d1 - d2)])
    return delta

def clip_rad_90(angle):
    while angle > np.pi/2:
        angle -=  np.pi
    while angle < -np.pi/2:
        angle +=  np.pi
    return angle

def clip_rad_0_90(angle):
    while angle > np.pi/2:
        angle -=  np.pi
    while angle < 0:
        angle +=  np.pi
    return angle

def clip_rad_180(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def clip_rad_360(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle

def convert_LP_to_GP(exp_global_position, exp_local_pose, door_pose):

    door_dist_to_start_pose = [int(door_pose[0] - exp_local_pose[0]), int(door_pose[1] - exp_local_pose[1])]
    entry_door_facing_rad = local_encoded_orientation_to_global_facing(0, exp_local_pose[2], exp_global_position[2])

    GP_door_pose = [0,0,0]
    GP_door_pose[0] = exp_global_position[0] +  (door_dist_to_start_pose[0] * np.cos(entry_door_facing_rad) - door_dist_to_start_pose[1] * np.sin(entry_door_facing_rad))
    GP_door_pose[1] = exp_global_position[1] +  (door_dist_to_start_pose[0] * np.sin(entry_door_facing_rad) + door_dist_to_start_pose[1] * np.cos(entry_door_facing_rad))
    GP_door_pose[2] = local_encoded_orientation_to_global_facing(door_pose[2], exp_local_pose[2], exp_global_position[2])

    return GP_door_pose
def local_encoded_orientation_to_global_facing(goal_lp_orientation:int,LP_start_orientation:int, GP_start_orientation:int)-> float:
    '''
    we have 4 orientations possible from 0 to 3 for the local pose in minigrid, 
    given the place local frame origin and corresponding global frame pose 
    we convert the orientation from one place local frame to global orientation in rad 
    '''
    #-- Option 2 --#
    orientations = [LP_start_orientation]
    GP_orientations = [GP_start_orientation]
    

    shifting_orientation = GP_start_orientation
    for i in range(1,4):
        orientations.append((LP_start_orientation+i)%4)
        shifting_orientation = shifting_orientation+np.pi/2
        GP_orientations.append(clip_rad_180(shifting_orientation))
    print('orientation and GP orientations', GP_orientations, orientations)
    goal_orientation_index = orientations.index(goal_lp_orientation)
    
    facing_rad = GP_orientations[goal_orientation_index]

    # -- OPTION1 -- #
    '''
    orientations= np.array([0,1,2,3] * 3)
    id_init_orientation = np.where(orientations == LP_start_orientation)[0][1]


    ids_desired_orientation = np.where(orientations == goal_lp_orientation)[0]
    closest_indexes = np.array(ids_desired_orientation)-id_init_orientation
    best_ids = np.where(abs(closest_indexes) == np.min(abs(closest_indexes)))[0]
    turn = closest_indexes[best_ids[0]]
    
    #print(best_ids)
    
    #180deg turn
    if len(best_ids) > 1 :
        facing_rad = clip_rad_180(GP_start_orientation  - (np.sign(GP_start_orientation) + int(GP_start_orientation == 0)) * np.pi)
    #forward
    elif turn == 0 :
        facing_rad = GP_start_orientation 
    #turning right
    elif turn > 0:
        #print('turning right')
        facing_rad = clip_rad_180(GP_start_orientation + np.pi / 2)
    #turning left
    else:
        #print('turning left')
        facing_rad = clip_rad_180(GP_start_orientation - np.pi / 2)
    '''
    return facing_rad

def encoded_orientation_given_facing(goal_GP_orientation, LP_start_orientation, GP_start_orientation):
    delta_angle = signed_delta_rad(GP_start_orientation, goal_GP_orientation)
    orientations = [LP_start_orientation]
    
    for i in range(1,4):
        orientations.append((LP_start_orientation+i)%4)

    orientation_shift = delta_angle/(np.pi/2)
    goal_lp_orientation = orientations[int(orientation_shift)]

    return goal_lp_orientation

def signed_delta_rad(angle1, angle2):
    dir = clip_rad_180(angle2 - angle1)

    delta_angle = abs(clip_rad_360(angle1) - clip_rad_360(angle2))

    if (delta_angle < (2 * np.pi - delta_angle)):
        if (dir > 0):
            angle = delta_angle
        else:
            angle = -delta_angle
    else:
        if (dir > 0):
            angle = 2 * np.pi - delta_angle
        else:
            angle = -(2 * np.pi - delta_angle)
    return angle

def sort_by_distance(exp, goal):
    #TODO: consider angle in a next iteration
    x,y,facing = exp['x'], exp['y'], exp['facing']
    xg,yg,facingg = goal['x'],goal['y'], goal['facing']
    delta_goal = np.sqrt(
            np.abs(x- xg)**2 +
            np.abs(y- yg)**2 
        )

    dist_angle = abs(signed_delta_rad(facing,facingg))
    delta_goal += dist_angle *1/(2*np.pi)  #to have the add_up < 1       
    exp['delta_goal'] = delta_goal #debug usage
    return delta_goal


def angle_between_two_poses(GP1:list, GP2:list) ->float:
    '''
    we calculate the angle between two vectors in a two-dimensional space defined 
    by their global positions (GP1 and GP2). It does this by finding the vector that 
    connects these two points and then determining the angle that this vector makes 
    with the horizontal axis. The result is the angular distance between the two vectors.
    '''
    vector_between_two_poses = np.subtract(GP1, GP2)
    angle_between_two_vectors = np.arctan2(vector_between_two_poses[1], vector_between_two_poses[0])
    return angle_between_two_vectors

def rad_distance_under_threshold(rad_diff:float, threshold:float=np.pi/2):
    return abs(rad_diff) < threshold