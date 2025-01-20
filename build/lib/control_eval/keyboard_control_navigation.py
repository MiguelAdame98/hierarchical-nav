#!/usr/bin/env python3

import os
import argparse
import numpy as np
import h5py
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from datetime import datetime

def redraw(img):
    #if not args.agent_view:
    img = env.render('rgb_array', tile_size=args.tile_size)
    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
   
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    print(action)
    obs, reward, done, info = env.step(action)
    print('step=%s' % (env.step_count))
    # print(obs) # action 6dof + image rgb 56,56,3
    # print(obs['image'].shape)
    data_dictionary['image'].append(obs['image'])
    data_dictionary['action'].append(obs['action'])
    data_dictionary['vel_ob'].append(obs['vel_ob'])
    #TODO: save the elements in a csv file or .h5 doc

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=True,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
try:
    data_dictionary = {'image':[],'vel_ob':[],'action':[]}
    now = datetime.now()
    current_time = now.strftime("%H-%M-%d-%m-%y")

    args = parser.parse_args()
    
    env = gym.make(args.env)

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgActionObsWrapper(env)
    # else:
    #     env = OneHotPartialObsWrapper(env)    

    window = Window('gym_minigrid - ' + args.env)
    window.reg_key_handler(key_handler)

    reset()

    # Blocking event loop
    window.show(block=True)
finally:
    cwd = os.getcwd()
    if not os.path.exists(cwd + '/data'):
        os.makedirs(os.path.join(cwd, 'data'), exist_ok=True)
        
    h5file = h5py.File(cwd +'/data/data'+ current_time+ '.h5', 'w')
    for key in data_dictionary.keys():
        h5file.create_dataset(
                key, data=data_dictionary[key], compression='lzf')
        print(h5file.get(key))
    h5file.close()
    print("Saving data as: data" + current_time + '.h5 in data folder' )
