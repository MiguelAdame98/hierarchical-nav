## Description

This repository is the source code of the 3 layered hierarchical model.

Information can be found on our Smart Robot blog [Hierarchy to navigate](https://thesmartrobot.github.io/2023/09/25/Spatial-Tempo-Hactinf.html)
or in our [article]( 	
https://doi.org/10.48550/arXiv.2312.05058)


Basic usage shows how to run the code for a demonstration. 
This condensed version has not been tested for a full training. 

## Basic Usage

To run a simulation in minigrid maze environments:
```
python run.py --env 4-tiles-ad-rooms --seed 218 --rooms_in_row 3 --rooms_in_col 4 --test key --video
```
<p align="center">
  <img src="img/3x4_s218.png" width="300" title="3x4 env seed 218">
  
</p>
#==== ENV related arguments ====# 

`--env`(str): Select the desired environment among
```
4-tiles-ad-rooms
5-tiles-ad-rooms
6-tiles-ad-rooms
7-tiles-ad-rooms
8-tiles-ad-rooms
```
The environments are generated randomly (rooms colours, configuration and white tile(s) positions)\
`--seed`(int): Between -1 and max system's value to select the environment generated. -1 will generate a new random configuration at each new call.\
`--rooms_in_row`(int): How many rooms we want to generate on a row\
`--rooms_in_col`(int): How many rooms we want to generate on a col

#==== Model arguments ====#\
`--allo_config`(str): path to the allocentric model to load as yaml\
`--memory_config`(str): path to the memory_graph_config as yaml\
`--memory_load`(str):path to a .map memory, None by default, the agent starts without prior on teh environment\
`--lookahead`(int): how many discrete steps ahead does the agent projects its imagination 

#==== TESTS arguments ====#\
`--video`: if we have this argument, we record what the agent is doing at each step as well as what is imagined.\
`--save_dir`(str): where we want to save dir. NOTE: currently adapted for a personal usage (not generalised), can be modified in input_output.py\
`--test`(str): Select the desired test run among :
```
key : Manual keyboard control of the agent navigation (we have a direct visual of the env) 
exploration: The agent autonomously aims to explore all the rooms of the environment. 
goal: The agent autonomously aims to reach a white tile 
```
The autonomous tasks can be agenced together such as 'exploration_then_goal'
NOTE: For the Goal, Another colour can be set, currently in code in keyboard_control_navigation as `preferred_colour_range`, However the Oracle currently search for a 'Goal' like tile to identify the goal. It can be adapted in Oracle or in Minigrid Goal setup colour.


The main.py is used to train and evaluate the models. 
However it might not be usable as it is because of migration issue. The usual usage is meant to be:

```
python3 main.py evaluate experiments/GQN_v2/GQN.py config=xxxx/GQN.yml learn_steps=20 n_sample=3 lookahead=20 lookahead_ratio=False unique_csv=Truev show_img=False save_img_seq=True scenario=xxx
```
# Requirements
Please run the requirements file 

To install locally:

git clone https://github.com/my-name-is-D/gym_minigrid_minimal.git
cd /your_repo/gym_minigrid_minimal && \
pip install -e . 

To install on cluster:
git clone -b master https://$DML_DEPLOY_TOKEN:$DML_DEPLOY_KEY@gitlab.ilabt.imec.be/ddtinguy/hierarchical_st_nav_aif.git && \
cd hierarchical_st_nav_aif && \
pip install -e .

# Environments

Currently strict minimum in repo, would be great to have a submodule instead (once disponible)

# Minimalistic Gridworld Environment (MiniGrid)

[![Build Status](https://travis-ci.org/maximecb/gym-minigrid.svg?branch=master)](https://travis-ci.org/maximecb/gym-minigrid)

There are other gridworld Gym environments out there, but this one is
designed to be particularly simple, lightweight and fast. The code has very few
dependencies, making it less likely to break or fail to install. It loads no
external sprites/textures, and it can run at up to 5000 FPS on a Core i7
laptop, which means you can run your experiments faster. A known-working RL
implementation can be found [in this repository](https://github.com/lcswillems/torch-rl).

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Matplotlib (optional, only needed for display)

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{gym_minigrid,
  author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
  title = {Minimalistic Gridworld Environment for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maximecb/gym-minigrid}},
}
```

List of publications & submissions using MiniGrid or BabyAI (please open a pull request to add missing entries):
- [In a Nutshell, the Human Asked for This: Latent Goals for Following Temporal Specifications](https://openreview.net/pdf?id=rUwm9wCjURV) (Imperial College London, ICLR 2022)
- [Interesting Object, Curious Agent: Learning Task-Agnostic Exploration](https://arxiv.org/abs/2111.13119) (Meta AI Research, NeurIPS 2021)
- [Safe Policy Optimization with Local Generalized Linear Function Approximations](https://arxiv.org/abs/2111.04894) (IBM Research, Tsinghua University, NeurIPS 2021)
- [A Consciousness-Inspired Planning Agent for Model-Based Reinforcement Learning](https://arxiv.org/abs/2106.02097) (Mila, McGill University, 2021)
- [SPOTTER: Extending Symbolic Planning Operators through Targeted Reinforcement Learning](http://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1118.pdf) (Tufts University, SIFT, AAMAS 2021)
- [Grid-to-Graph: Flexible Spatial Relational Inductive Biases for Reinforcement Learning](https://arxiv.org/abs/2102.04220) (UCL, AAMAS 2021)
- [Rank the Episodes: A Simple Approach for Exploration in Procedurally-Generated Environments](https://openreview.net/forum?id=MtEE0CktZht) (Texas A&M University, Kuai Inc., ICLR 2021)
- [Adversarially Guided Actor-Critic](https://openreview.net/forum?id=_mQp5cr_iNy) (INRIA, Google Brain, ICLR 2021)
- [Information-theoretic Task Selection for Meta-Reinforcement Learning](https://papers.nips.cc/paper/2020/file/ec3183a7f107d1b8dbb90cb3c01ea7d5-Paper.pdf) (University of Leeds, NeurIPS 2020)
- [BeBold: Exploration Beyond the Boundary of Explored Regions](https://arxiv.org/pdf/2012.08621.pdf) (UCB, December 2020)
- [Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems](https://arxiv.org/abs/2010.08843) (McGill, October 2020)
- [Prioritized Level Replay](https://arxiv.org/pdf/2010.03934.pdf) (FAIR, October 2020)
- [AllenAct: A Framework for Embodied AI Research](https://arxiv.org/pdf/2008.12760.pdf) (Allen Institute for AI, August 2020)
- [Learning with AMIGO: Adversarially Motivated Intrinsic Goals](https://arxiv.org/pdf/2006.12122.pdf) (MIT, FAIR, ICLR 2021)
- [RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://openreview.net/forum?id=rkg-TJBFPB) (FAIR, ICLR 2020)
- [Learning to Request Guidance in Emergent Communication](https://arxiv.org/pdf/1912.05525.pdf) (University of Amsterdam, Dec 2019)
- [Working Memory Graphs](https://arxiv.org/abs/1911.07141) (MSR, Nov 2019)
- [Fast Task-Adaptation for Tasks Labeled Using
Natural Language in Reinforcement Learning](https://arxiv.org/pdf/1910.04040.pdf) (Oct 2019, University of Antwerp)
- [Generalization in Reinforcement Learning with Selective Noise Injection and Information Bottleneck
](https://arxiv.org/abs/1910.12911) (MSR, NeurIPS, Oct 2019)
- [Recurrent Independent Mechanisms](https://arxiv.org/pdf/1909.10893.pdf) (Mila, Sept 2019) 
- [Learning Effective Subgoals with Multi-Task Hierarchical Reinforcement Learning](http://surl.tirl.info/proceedings/SURL-2019_paper_10.pdf) (Tsinghua University, August 2019)
- [Mastering emergent language: learning to guide in simulated navigation](https://arxiv.org/abs/1908.05135) (University of Amsterdam, Aug 2019)
- [Transfer Learning by Modeling a Distribution over Policies](https://arxiv.org/abs/1906.03574) (Mila, June 2019)
- [Reinforcement Learning with Competitive Ensembles
of Information-Constrained Primitives](https://arxiv.org/abs/1906.10667) (Mila, June 2019)
- [Learning distant cause and effect using only local and immediate credit assignment](https://arxiv.org/abs/1905.11589) (Incubator 491, May 2019)
- [Practical Open-Loop Optimistic Planning](https://arxiv.org/abs/1904.04700) (INRIA, April 2019)
- [Learning World Graphs to Accelerate Hierarchical Reinforcement Learning](https://arxiv.org/abs/1907.00664) (Salesforce Research, 2019)
- [Variational State Encoding as Intrinsic Motivation in Reinforcement Learning](https://mila.quebec/wp-content/uploads/2019/05/WebPage.pdf) (Mila, TARL 2019)
- [Unsupervised Discovery of Decision States Through Intrinsic Control](https://tarl2019.github.io/assets/papers/modhe2019unsupervised.pdf) (Georgia Tech, TARL 2019)
- [Modeling the Long Term Future in Model-Based Reinforcement Learning](https://openreview.net/forum?id=SkgQBn0cF7) (Mila, ICLR 2019)
- [Unifying Ensemble Methods for Q-learning via Social Choice Theory](https://arxiv.org/pdf/1902.10646.pdf) (Max Planck Institute, Feb 2019)
- [Planning Beyond The Sensing Horizon Using a Learned Context](https://personalrobotics.cs.washington.edu/workshops/mlmp2018/assets/docs/18_CameraReadySubmission.pdf) (MLMP@IROS, 2018)
- [Guiding Policies with Language via Meta-Learning](https://arxiv.org/abs/1811.07882) (UC Berkeley, Nov 2018)
- [On the Complexity of Exploration in Goal-Driven Navigation](https://arxiv.org/abs/1811.06889) (CMU, NeurIPS, Nov 2018)
- [Transfer and Exploration via the Information Bottleneck](https://openreview.net/forum?id=rJg8yhAqKm) (Mila, Nov 2018)
- [Creating safer reward functions for reinforcement learning agents in the gridworld](https://gupea.ub.gu.se/bitstream/2077/62445/1/gupea_2077_62445_1.pdf) (University of Gothenburg, 2018)
- [BabyAI: First Steps Towards Grounded Language Learning With a Human In the Loop](https://arxiv.org/abs/1810.08272) (Mila, ICLR, Oct 2018)

This environment has been built as part of work done at [Mila](https://mila.quebec). The Dynamic obstacles environment has been added as part of work done at [IAS in TU Darmstadt](https://www.ias.informatik.tu-darmstadt.de/) and the University of Genoa for mobile robot navigation with dynamic obstacles.


## Wrappers

MiniGrid is built to support tasks involving natural language and sparse rewards.
The observations are dictionaries, with an 'image' field, partially observable
view of the environment, a 'mission' field which is a textual string
describing the objective the agent should reach to get a reward, and a 'direction'
field which can be used as an optional compass. Using dictionaries makes it
easy for you to add additional information to observations
if you need to, without having to encode everything into a single tensor.

There are a variery of wrappers to change the observation format available in [gym_minigrid/wrappers.py](/gym_minigrid/wrappers.py). If your RL code expects one single tensor for observations, take a look at
`FlatObsWrapper`. There is also an `ImgObsWrapper` that gets rid of the 'mission' field in observations,
leaving only the image field tensor.

Please note that the default observation format is a partially observable view of the environment using a
compact and efficient encoding, with 3 input values per visible grid cell, 7x7x3 values total.
These values are **not pixels**. If you want to obtain an array of RGB pixels as observations instead,
use the `RGBImgPartialObsWrapper`. You can use it as follows:

```
from gym_minigrid.wrappers import *
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs = env.reset() # This now produces an RGB tensor only
```

## Design

Structure of the world:
- The world is an NxM grid of tiles
- Each tile in the grid world contains zero or one object
  - Cells that do not contain an object have the value `None`
- Each object has an associated discrete color (string)
- Each object has an associated type (string)
  - Provided object types are: wall, floor, lava, door, key, ball, box and goal
- The agent can pick up and carry exactly one object (eg: ball or key)
- To open a locked door, the agent has to be carrying a key matching the door's color

Actions in the basic environment:
- Turn left
- Turn right
- Move forward
- Pick up an object
- Drop the object being carried
- Toggle (open doors, interact with objects)
- Done (task completed, optional)

Default tile/observation encoding:
- Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE) 
- OBJECT_TO_IDX and COLOR_TO_IDX mapping can be found in [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
- e.g. door STATE -> 0: open, 1: closed, 2: locked

By default, sparse rewards are given for reaching a green goal tile. A
reward of 1 is given for success, and zero for failure. There is also an
environment-specific time step limit for completing the task.
You can define your own reward function by creating a class derived
from `MiniGridEnv`. Extending the environment with new object types or new actions
should be very easy. If you wish to do this, you should take a look at the
[gym_minigrid/minigrid.py](gym_minigrid/minigrid.py) source file.

## Included Environments

The environments listed below are implemented in the [gym_minigrid/envs](/gym_minigrid/envs) directory.
Each environment provides one or more configurations registered with OpenAI gym. Each environment
is also programmatically tunable in terms of size/complexity, which is useful for curriculum learning
or to fine-tune difficulty.

### Four rooms environment

Registered configurations:
- `MiniGrid-FourRooms-v0`

<p align="center">
<img src="/figures/four-rooms-env.png" width=380>
</p>

Classic four room reinforcement learning environment. The agent must navigate
in a maze composed of four rooms interconnected by 4 gaps in the walls. To
obtain a reward, the agent must reach the green goal square. Both the agent
and the goal square are randomly placed in any of the four rooms.
