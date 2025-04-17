
import sys
from operator import itemgetter

import numpy as np
import torch

#--------------- ACTION PROCESSES ------------#
def action_to_pose(action,pose):
    current_pose = pose.copy()
    if isinstance(action, torch.Tensor) and len(action.shape)>1:
        action = action[0]
    # #GQN increment action whatever the real situation, so we check if no vel [0,0]
    # if vel_ob[0] ==vel_ob[1] :
    #     action = [0,0,0]
    DIR_TO_VEC = [
        # Pointing right (positive X)
        [1, 0],
        # Down (positive Y)
        [0, 1],
        # Pointing left (negative X)
        [-1, 0],
        # Up (negative Y)
        [0, -1],
    ]
    if action[0] == 1:
        current_pose[:2] = DIR_TO_VEC[current_pose[2]] + current_pose[:2]
    elif action[0] == -1:
        current_pose[:2] = [ -i for i in DIR_TO_VEC[current_pose[2]] ] + current_pose[:2]
    elif action[1] == 1:
        current_pose[2] = (current_pose[2]+1)%4
    elif action[2] == 1:
        current_pose[2] = (current_pose[2]-1)%4
    return current_pose

#--------------- POLICY PROCESSES ------------#
def define_policies_objectives(current_pose:list, lookahead:int, full_exploration:bool) ->list:
    """ 
    we determine if we want the agent to just reach forward or a full 2D exploration
    around the agent. 
    If full_exploration is True : all corners of square (dist to agent:lookahead) 
    perimeters around agent set as goal
    """
    goal_poses = []

    goal_poses.append([current_pose[0]+lookahead,  current_pose[1]-lookahead, current_pose[2]])
    goal_poses.append([current_pose[0]+lookahead,  current_pose[1]+lookahead, current_pose[2]])
    
    # if we have a stable place modelised, then we can explore the negative positions of the newly created place
    if full_exploration :
        goal_poses.append([current_pose[0]-lookahead,  current_pose[1]-lookahead, current_pose[2]])
        goal_poses.append([current_pose[0]-lookahead,  current_pose[1]+lookahead, current_pose[2]])
    
    return goal_poses

def create_policies(current_pose:list, goal_poses:list, exploration:bool=True)-> list:
    ''' Given current pose, and the goals poses
    we want to explore or reach those goals, 
    generate policies going around in a square perimeter. 
    Either just forward (goals), or all around (explore)'''

    policies_lists = []
    #get all the actions leading to the endpoints
    for endpoint in goal_poses:
        print('end point and current pose', endpoint, current_pose)
        action_seq_options = define_policies_to_goal(current_pose, endpoint, exploration)
        policies_lists.extend(action_seq_options)

    return policies_lists

def exploration_goal_square(dx, dy):
    '''
    Create 1 path going to given dx for each y latitude and vice versa for dy. 
    This limit the path generation to half (opposing sides of the starting agent position) 
    the number of tiles on the outline of the rectangle formed by dx and dy.
    LIMITATION: WORKS WITH squarred areas
    '''
    paths = [[[0,0]]]
    paths.append([[0,0]])
    while paths[-1][-1] != [dx,dy] :
        new_paths = []
        #print('last path step', paths[-1][-1])
        for p in range(0,len(paths),2):
            #print(p,paths)
            #print('p',p, paths[p][-1], paths[p+1][-1], dx, dy)
            try:
                if paths[p+1][-1][1] < dy:
                    new_step_y = paths[p+1][-1].copy()
                    new_step_y[1] += 1
                    paths[p+1].append(new_step_y)
                    if p == 0:
                        new_paths.append(paths[p+1].copy())
            #this means we need to change axis more on x than y
            except IndexError:
                pass
            try:
                if paths[p][-1][0] < dx:
                    new_step_x = paths[p][-1].copy()
                    new_step_x[0] += 1
                    paths[p].append(new_step_x)
                    if p == 0:
                        new_paths.append(paths[p].copy())
            #this means we need to change axis more on y than x
            except IndexError:
                pass
            #print('end for ', p, ' paths', paths)
        paths.extend(new_paths)

    return paths

def two_paths_to_goal(dx,dy):
    '''
    Output the 2 shortest paths leading to goal, 
    so it's the rectangle outline between (0,0) and (dx,dy)
    '''
    paths = [[[0,0]]]
    paths.append([[0,0]])
    for p in range(2):
        while paths[p][-1][0] < dx:
            new_step_x = paths[p][-1].copy()
            new_step_x[0] += 1
            paths[p].append(new_step_x)

        while paths[-1-p][-1][1] < dy:
            new_step_y = paths[-1-p][-1].copy()
            new_step_y[1] += 1
            paths[-1-p].append(new_step_y)
        #print('p paths', p, paths)
    return paths

#TODO: SIMPLIFY STP
def define_policies_to_goal(start_pose:list, end_pose:list, exploration:bool= True)->list:
    '''
    MINIRID ADAPTED AS ORIENTATION ARE ONLY 4 (90deg turns)
    Given the current pose and goal pose establish all the sequence of actions 
    leading TOWARD the objective. Do not propose all turn options if needed to turn 
    180deg at start and end. 
    This code is only valid in a zone without obstacles in room. If there are, consider
    expanding the area of possible paths.
    '''
    #NOTE: THIS IS CONSIDERING POSES WITH 2 POSES IN SAME ROOM
    #we shift from the 0-3 orientation representation to 1-4
    
    start_theta, end_theta = start_pose[2]+1,  end_pose[2]+1
    dx,dy = abs(int(start_pose[0] - end_pose[0])), abs(int(start_pose[1] - end_pose[1])) # destination cell

    #If we want to reach a goal, we just need the 2 shortest paths leading there
    if not exploration:
        paths = two_paths_to_goal(dx, dy)  
    #If we want to explore, we want a grid path coverage (squared)
    else:    
        paths = exploration_goal_square(dx, dy)
        
    action_seq_options = []
    orientations= np.array([1,2,3,4] * 3)
   
    for path in paths:
        path = np.array(path)
        if start_pose[0] > end_pose[0]:
            path[:,0]= -path[:,0]
        if start_pose[1] > end_pose[1]:
            path[:,1]= -path[:,1]
            
        path = path.tolist()
        action_seq = []
        action_seq_alt = []
        for step in range(1,len(path)+1):
            # print('step', step)
            continue_path = True
            #first step init, add orientation
            if step == 1:
                path[step-1].append(start_theta)
            
            #last step final orientation check
            if step == len(path):
                path.append(path[-1])
                path[step].append(end_theta)
                desired_orientation = end_theta
                continue_path = False
            
            if path[step][0] - path[step-1][0] > 0 : #go forward x 
                desired_orientation = 1
            elif path[step][0] - path[step-1][0] < 0 : #go backward x
                desired_orientation = 3
            elif path[step][1] - path[step-1][1] > 0 : #go forward y
                desired_orientation = 2
            elif path[step][1] - path[step-1][1] < 0 : #go backward y
                desired_orientation = 4
            
            path[step].append(desired_orientation)

            # --- check if we need to turn and apply action --- #
            id_current_orientation = np.where(orientations == path[step-1][2])[0][1]
            ids_desired_orientation = np.where(orientations == desired_orientation)[0]
            closest_indexes = np.array(ids_desired_orientation)-id_current_orientation

            best_ids = np.where(abs(closest_indexes) == np.min(abs(closest_indexes)))[0]

            #NOTE: we can save alternative paths by uncommenting the part below, 
            # commented because of computation issue (only 1 cpu core used)
            for option, id in enumerate(best_ids):
                seq = []
                # print('best ids',best_ids, closest_indexes)
                n_turn = closest_indexes[id]
                # print('number of turn', n_turn)
                if n_turn < 0:
                    turn = [0,0,1] #turn left
                elif n_turn > 0:
                    turn = [0,1,0] #turn right
                for i in range(abs(n_turn)):
                    seq.append(turn) # turn 
                
                if continue_path:
                    seq.append([1,0,0])

                #NOTE: there is only 1 alternative list, so won't consider ALL 180deg turns. 
                # Meaning that if 2 are necessary at start and end, you won't 
                # have RR--RR  / RR--LL / LL--LL / LL--RR 
                # but only RR--RR / LL--LL 
                
                #--- Save action in action_seq or alternative_action_seq --#
                if option ==0 :
                    ##If we have no alternative action seq but 2 possible motions, get all prev actions 
                    if action_seq_alt == [] and len(best_ids) > 1:
                            action_seq_alt = action_seq.copy()
                    ##If we have an alternative action seq and we are on an unique path 
                    elif action_seq_alt != [] and len(best_ids) == 1:
                         action_seq_alt.extend(seq)
                    action_seq.extend(seq)
                #update the second motion option in the alternative seq 
                else:
                     action_seq_alt.extend(seq)
        #path_lengths.append(len(action_seq))
        action_seq_options.append(action_seq)
        if action_seq_alt != []:
             
        #     # print('added', action_seq)
        #     # print('added', action_seq_alt)
                action_seq_options.append(action_seq_alt)
        #path_lengths.append(len(action_seq_alt))
    
    return action_seq_options

#--------------- DIJKSTRA PROCESSES ------------#
def dijkstra_weigthed(exps_list:list, start_id:int, proba_impact:int = 100)->list:
        '''
        Dijkstra's shortest path, considering node dist + how many node to pass from
        The number of node to pass by is more important than the distance between nodes
        Djikstra check through all exps. A High proba_impact means that unlinked exps will
        have a low proba of being considered, A low proba_impact means that ghost nodes and
        unconnected nodes gets considered.
        '''
        # Put tuple pair into the priority queue
        unvisited_queue = []
        
        
        for exp in exps_list:
            init_setting = {'weight':sys.maxsize, 'num_nodes': sys.maxsize, 'previous_exp':None, 
                        'visited_exp':False, 'linked_exps': [], 'num_nodes':0} #IN FOR to avoid python shared memory issues with nested elements
            exp.update(init_setting)
            # Set the distance for the start node to zero 
            if exp['id'] == start_id:
                exp['weight'] = 0
            
            for link in exp['links']:
                exp['linked_exps'].append(link.target.id)

            unvisited_queue.append((exp['weight'],exp))

        while len(unvisited_queue) > 0:
            # Pops exp with the smallest distance 
            unvisited_queue = sorted(unvisited_queue, key=itemgetter(0))
            uv = unvisited_queue.pop(0)
            current = uv[1]
            current['visited_exp'] = True

            #for next linked exp (only go through links)
            # for next_exp_id in current['linked_exps']:
            #     exp_index_in_list = [exp['id'] for exp in exps_list].index(next_exp_id)
            #     next = exps_list[exp_index_in_list]
            
            #for next among ALL nodes:
            for next in exps_list :
                if next['id'] == current['id']:
                    continue
                next_exp_id = next['id']

                if next_exp_id in current['linked_exps']:
                    node_dist, ghost_exp = get_linked_exps_dist(current['links'],next_exp_id)
                    if node_dist is None and ghost_exp is None:
                        continue
                    if not ghost_exp:
                        exps_linked_proba = 1
                    else:
                        exps_linked_proba = 1 * 4 / (proba_impact)
                
                else:
                    node_dist = get_exps_euclidian_dist(current,next)
                    exps_linked_proba = 1/ proba_impact
                
                # if visited, skip
                if next['visited_exp']:
                    continue
                new_weight = current['weight'] + node_dist + (node_dist * ((1-exps_linked_proba) * proba_impact))
                
                #If new weight < to the next node currently calculated weigth 
                # OR if there are less nodes to go there
                #Then we replace next weigth by current and link exps
                if new_weight < next['weight'] or (current['num_nodes'] + 1 < next['num_nodes']):
                    next['weight'] = new_weight
                    next['num_nodes'] = current['num_nodes'] + 1
                    next['previous_exp'] = current
                    print('updated : current = %s next = %s new_weight = %s num_nodes = %s' \
                            %(current['id'], next['id'], next['weight'], next['num_nodes']))
                else:
                    print('not updated : current = %s next = %s next weight = %s , new_weight = %s , num_nodes = %s' \
                            %((current['id'], next['id'], next['weight'], new_weight, next['num_nodes'])))
                # print()
            # Clear used unvisited queue
            unvisited_queue.clear()
            # Fill the queue with all non visited nodes for next iteration
            for exp in exps_list:
                if not exp['visited_exp']: 
                    unvisited_queue.append((exp['weight'],exp))
            
        return exps_list

def dijkstra_shortest_node_path(target_exp:dict, path:list)->list:
    ''' make shortest path from linked previous_exp.
        Start from the target exp.
    '''
    if target_exp['previous_exp'] is not None:
        path.append(target_exp['previous_exp']['id'])
        path = dijkstra_shortest_node_path(target_exp['previous_exp'], path)
    return path

def get_linked_exps_dist(links:list, neighbor_id:int):
    for link in links:
        if link.target.id == neighbor_id:
            return link.d, link.target.ghost_exp  
    return None, None

#--------------- EUCLIDIAN DIST PROCESSES ------------#
def get_exps_euclidian_dist(exp1:dict, exp2:dict) -> float:
    delta_exps = np.sqrt(
        np.abs(exp1['x'] - exp2['x'])**2 +
        np.abs(exp1['y'] - exp2['y'])**2 
    )
    return delta_exps

def get_pose_euclidian_dist(p1:list, p2:list)->float:
    delta_poses = np.sqrt(
        np.abs(p1[0] - p2[0])**2 +
        np.abs(p1[1] - p2[1])**2 
    )
    return delta_poses
