import io
import logging
import os
import time
from pathlib import Path

import h5py as h5
import torch
import yaml
from tqdm import tqdm

try:
    from pynput.keyboard import Controller, Key
except:
    pass
import csv


def load_h5(filename):
    """
    :param filename: Location to load data from (<path/to/file.h5>)
    """
    data = {}
    with h5.File(filename, "r") as f:
        for key in f.keys():
            data[key] = torch.tensor(f[key][:], dtype=torch.float32)
    return data


def press_keyboard(key:str = 'Key.enter')-> None:
        keyboard = Controller()     
        time.sleep(0.5)   
        keyboard.press(eval(key))
        time.sleep(0.005)
        keyboard.release(eval(key)) 

def setup_allocentric_config(allo_config:str)->dict:
    try:
        allo_config = yaml.safe_load(Path(allo_config).read_text())
    except Exception as e:
        raise Exception(e)
    allo_config['params'] = allo_config['log_dir']
    return allo_config

def setup_memory_config(memory_config:str)-> dict:
    try:
        memory_config = yaml.safe_load(Path(memory_config).read_text())
    except Exception as e:
        raise Exception(e)
    return memory_config

def load_memory(models_manager:object, old_memory:str) -> None:
    '''
    We re-use previously memorised experiences
    '''
    if old_memory is not None:
        try:
            models_manager.load_memory(old_memory)
        except Exception as e:
            raise Exception(e)

def save_memory(models_manager:object, saving_directory:str)-> None:
    """
    Save the memorised map in the desired directory
    """
    models_manager.save_memorised_map(saving_directory + '.map')

#TODO:MAKE THIS MORE MODULAR
def create_saving_directory(directory:str)-> str:
    """NOTE TIS IS PARTICULAR TO ONE INDIVIDUAL"""
    if os.path.exists('/Users/lab25'):
        dir = '/Users/lab25/Documents/hierarchical_st_nav_aif/' 
    else:
        dir = '/Users/lab25/hierarchical_st_nav_aif/' 

    home_dir = dir + directory 
           
    create_directory(home_dir)

    return home_dir

def create_directory(directory:str)->bool:
    try:
        os.makedirs(directory)
        return True
    except FileExistsError:
        return False
    
def setup_video(saving_dir, env_details, record_video, run_test):
    import imageio
    if record_video:
        count = 0
        video_file = saving_dir + '/'+ run_test +'_'+env_details+'_'+str(count)+ ".mp4"
        while os.path.exists(video_file):
            count += 1
            video_file = saving_dir + '/'+ run_test +'_'+env_details+'_'+str(count)+ ".mp4"
        video_gridmap = imageio.get_writer(video_file, format='FFMPEG', fps=1)
    else:
        video_gridmap = False
    return video_gridmap

def get_user_input():
    user_pose = []
    for i in range(3):
        element = input(f"Enter element axis {i + 1}: ")
        user_pose.append(int(element))

    return user_pose
    
def save_experiment_data(data, env_details, test_type, save_dir):
    from collections import OrderedDict

    import numpy as np
    
    csv_name = save_dir + '/' + test_type +'_exps_results.csv'
    
    
    data['env'] = env_details
    # data['n_exps'], _, data['map_coverage'] = navigation_state
    # data['per_map_coverage'] = data['map_coverage']/max_visible_tiles
    
    
    if 'goal' in test_type:
        goal_diff_steps = np.subtract(data['goal_in_map'][:2],data['last_p_agent_in_map'][:2])
        data['agent_dist_goal'] = sum(abs(number) for number in goal_diff_steps)

        data['n_diff_goal_steps'] = data['steps_to_goal'] - data['oracle_steps_to_goal']

        # data['agent on goal'] = data['goal_in_map'] == data['last_p_agent_in_map'][:2]
        # header.extend(['goal_reached?', \
        #         'oracle_steps_to_goal', 'steps_to_goal', 'goal_in_map',  'last_p_agent_in_map', \
        #         'agent on goal', 'agent_dist_goal', 'n_diff_goal_steps' ])

    if 'exploration' in test_type:
        data['n_diff_explo_steps'] = data['exploration_n_steps'] - data['oracle_exploration'] 
        data['n_visited_rooms']= len(data['visited_rooms'])
    #to be sure it's always well ordered
    data = OrderedDict(sorted(data.items(), key=lambda t: t[0]))
    info,header = [], []
    print('in save_experiment_data')
    for key, value in data.items():
        print(key, value, type(value))
        header.append(key)
        info.append(value)
   
    print('header', header)
    print('info', info)
    #If we don't want to print the header each time (necessary to always save the same values type for 1 csv file)
    if not os.path.exists(csv_name):
        no_header = True
    else:
        no_header = False

    with open(csv_name, 'a+', encoding="UTF8") as file:
        writer = csv.writer(file)
        if no_header:
            writer.writerow(header)
        writer.writerow(info)


#================================ LOGS ==================================================
class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

def set_log():
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    tqdm_out = TqdmToLogger(logger,level=logging.INFO)
    return tqdm_out
