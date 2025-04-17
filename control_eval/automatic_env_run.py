import logging
import traceback

from control_eval.input_output import (create_saving_directory,
                                       save_experiment_data, setup_video)
from Oracle_Astar_algo import Oracle_path
from navigation_model.visualisation_tools import record_video_frames

def env_definition(args): 
    ''' extract most relevant flags, just for order sake'''   
    env_room_size = int(args.env[0])
    env_row = args.rooms_in_row
    env_col = args.rooms_in_col
    max_steps = env_row * env_col * env_room_size *18
    
    env_definition = {'room_size': env_room_size, 'n_row': env_row, \
                    'n_col': env_col, 'max_steps':max_steps,\
                    'seed':args.seed}
        
    return env_definition

def run_test(step_controller, flags):
    env_def = env_definition(flags)
    env_details = str(env_def['room_size']) +'t_'+ str(env_def['n_row'])+'x'+str(env_def['n_col']) + '_s'+ str(env_def['seed'])
    home_saving_dir = create_saving_directory(flags.save_dir)
    exp_saving_dir = home_saving_dir + '/' + env_details   
    video_gridmap = setup_video(exp_saving_dir, env_details, flags.video, flags.test)
    final_data = {}
    final_data['Error_occured?'] = False
    oracle = Oracle_path(step_controller.env)
    
    try:
        if 'exploration'in flags.test :
            explo_path, final_data['oracle_exploration'] = oracle.steps_to_explore_env()
            print(explo_path, final_data['oracle_exploration'])
            final_data['exploration_success?'], final_data['visited_rooms'], video_gridmap, final_data['Error_occured?'] \
                = run_exploration(step_controller, env_def, oracle, video_gridmap)
            final_data['exploration_n_steps'] = step_controller.step_count()
            step_controller.save_created_map(exp_saving_dir + '/map_' + env_details)
        if 'goal' in flags.test:
            goal_path, final_data['oracle_steps_to_goal'] = oracle.steps_to_closest_goal()
            final_data['goal_in_map'] = list(oracle.goal_position[0])
            start_step = step_controller.step_count()
            final_data['goal_reached?'], final_data['goal_rooms_path'], video_gridmap, final_data['Error_occured?'] \
                = run_goal_seeking(step_controller, env_def, oracle, video_gridmap, final_data['goal_in_map'])
            final_data['steps_to_goal'] = step_controller.step_count() - start_step
            
    except:
        error_message = traceback.format_exc() 
        print('ERROR MESSAGE:',error_message)
        print('EXPERIMENT INTERRUPTED')  
        final_data['Error_occured?'] = True   
    
    finally:
        if video_gridmap:
            video_gridmap.close()
        final_data['last_p_agent_in_map'] = step_controller.agent_absolute_pose()
        save_experiment_data(final_data, env_details, flags.test, home_saving_dir)
        print('EXPERIMENT DONE')
        print('++++++++++++++++++++++++++++++++++++++++++++')
#============================== GOAL METHODS ============================================

def run_goal_seeking(step_controller:object, env_definition:dict, oracle:object, video_gridmap:object, goal_in_map:list):
    rooms_limits = oracle_rooms_limit(oracle.get_doors_poses_in_env())
    visited_rooms = [] 
    goal_reached= False
    
    prev_log_step = 0
    #Just to be sure we have a memory ready to use for goal search
    while step_controller.agent_situate_memory() < 0:   
        if step_controller.step_count() <= 1:     
            action = convert_string_to_env_action('f')
            step_controller.step(action)
        else:
            steps = step_controller.step_count()
            motion_data, agent_lost = step_controller.agent_explores(collect_data=True)
            visited_rooms = count_visited_rooms(step_controller.agent_absolute_pose(), rooms_limits, visited_rooms)
            video_gridmap = record_data(step_controller,video_gridmap, motion_data, env_definition, agent_lost, [visited_rooms], steps)
    print('Agent ready to search for goal')
    steps = step_controller.step_count()
    try:
        while steps <= env_definition['max_steps']:
            motion_data, agent_lost = step_controller.agent_seeks_goal(collect_data=True)
            agent_in_map = step_controller.agent_absolute_pose()
            visited_rooms = count_visited_rooms(agent_in_map, rooms_limits, visited_rooms)
            video_gridmap = record_data(step_controller,video_gridmap, motion_data, env_definition, agent_lost, visited_rooms, steps)
            
            steps = step_controller.step_count()
            if steps - prev_log_step >= 25:
                    logging.info('Goal seeking step:'+ str(steps)+'/'+ str(env_definition['max_steps']) + ','+str(len(visited_rooms)) + ' passed rooms up to now')
                    prev_log_step = steps
            goal_reached = goal_condition_filled(goal_in_map, agent_in_map)
            if goal_reached:
                print('The agent found the goal. GOAL REACHED')
                if video_gridmap:
                    #This is just because I can't see the last frame if i don't add a delay
                    memory_map_data = step_controller.get_memory_map_data()
                    frame = record_video_frames(motion_data[-1], env_definition, agent_lost, visited_rooms, memory_map_data, steps)
                    video_gridmap.append_data(frame)
                    frame = record_video_frames(motion_data[-1], env_definition, agent_lost, visited_rooms, memory_map_data, steps)
                    video_gridmap.append_data(frame)
                break

        return goal_reached, visited_rooms, video_gridmap, False

    except:
            error_message = traceback.format_exc() 
            print('ERROR MESSAGE:',error_message) #Both so that can be printed in txt
            logging.error(str(error_message))
            print('EXPERIMENT INTERRUPTED')     
            return goal_reached, visited_rooms, video_gridmap, True
    
def goal_condition_filled(goal_in_map, agent_im_map):
    """ are we on the goal position?"""
    if (goal_in_map[:2] == agent_im_map[:2]):
        return True
    return False
         
#============================== EXPLORATION METHODS =====================================
#TODO: Ideally it would be great to check if the agent has a wrong belief over the goal position and note it somewhere (or display the goal visual in the video)
def run_exploration(step_controller:object, env_definition:dict, oracle:object, video_gridmap:object):
    
        # tqdm_out = set_log() 
        rooms_limits = oracle_rooms_limit(oracle.get_doors_poses_in_env())
        print('rooms_limits', rooms_limits)
        visited_rooms = [] 
        full_exploration = False
        entered_last_room = 0

        action = convert_string_to_env_action('f')
        step_controller.step(action)
        
        steps = step_controller.step_count()
        prev_log_step = 0
        try:
            #while steps <= 2*(env_definition['max_steps']):
            while steps <= int(32):
                motion_data, agent_lost = step_controller.agent_explores(collect_data=True)
                visited_rooms = count_visited_rooms(step_controller.agent_absolute_pose(), rooms_limits, visited_rooms)
                video_gridmap = record_data(step_controller,video_gridmap, motion_data, env_definition, agent_lost, visited_rooms, steps)
                steps = step_controller.step_count()
                if steps - prev_log_step >= 25:
                        logging.info('Exploration step:'+ str(steps)+'/'+ str(env_definition['max_steps']) + ','+str(len(visited_rooms)) + ' visited rooms up to now')
                        prev_log_step = steps
                full_exploration = exploration_condition_filled(env_definition, visited_rooms)
                
                if full_exploration:
                    
                    print('all rooms have been visited')
                    if not step_controller.agent_lost() and entered_last_room > 2:
                        print('The agent got a grip of this last room. ENDING EXPLORATION')
                        if video_gridmap:
                            #This is just because I can't see the last frame if i don't add a delay
                            memory_map_data = step_controller.get_memory_map_data()
                            frame = record_video_frames(motion_data[-1], env_definition, agent_lost, visited_rooms, memory_map_data, steps)
                            video_gridmap.append_data(frame)
                            frame = record_video_frames(motion_data[-1], env_definition, agent_lost, visited_rooms, memory_map_data, steps)
                            video_gridmap.append_data(frame)
                        break
                    else:
                        entered_last_room += 1
        except:
            error_message = traceback.format_exc() 
            print('ERROR MESSAGE:',error_message) #Both so that can be printed in txt
            logging.error(str(error_message))
            print('EXPERIMENT INTERRUPTED')     
            return full_exploration, visited_rooms, video_gridmap, True
        
        return full_exploration, visited_rooms, video_gridmap, False

def exploration_condition_filled(env_definition:dict, visited_rooms:dict):
    num_rooms = env_definition['n_col'] * env_definition['n_row']
    return len(visited_rooms) == num_rooms
        
def count_visited_rooms(agent_pose:list, rooms_limits:dict, visited_rooms:list)->list:
    for room, dimensions in rooms_limits.items():
        
        if dimensions[0][0]  <= agent_pose[0] <= dimensions[0][1] and dimensions[1][0]  <= agent_pose[1] <= dimensions[1][1] :
            if room not in visited_rooms:
                print('exploration entering a new room', room)
                visited_rooms.append(room)
                break
    return visited_rooms

def oracle_rooms_limit(rooms_wt_door_poses:dict)-> dict:
    """Define the limits of each room"""
    rooms_limits = {}  
    
    for room in rooms_wt_door_poses.keys():
        rooms_limits[room] = [[min(rooms_wt_door_poses[room], key=lambda x: x[0])[0],max(rooms_wt_door_poses[room], key=lambda x: x[0])[0]],
                            [min(rooms_wt_door_poses[room], key=lambda x: x[1])[1], max(rooms_wt_door_poses[room], key=lambda x: x[1])[1]]]
    
    return rooms_limits


#================================ OTHER METHODS =========================================

def convert_string_to_env_action(action:str) -> int:
        if 'd' in action:
            action = 6
        if 'f' in action:
                action = 2
        elif 'r' in action:
            action = 1
        elif 'l' in action:
            action = 0
        else:
            raise 'unrecognised action to apply:'+str(action)
        return action

def record_data(step_controller,video_gridmap, motion_data, env_definition, agent_lost, visited_rooms, steps):
    if video_gridmap and motion_data: 
        memory_map_data = step_controller.get_memory_map_data()
        for data in motion_data:
            frame = record_video_frames(data, env_definition, agent_lost, visited_rooms, memory_map_data, steps)
            video_gridmap.append_data(frame)
            steps+= 1
    return video_gridmap