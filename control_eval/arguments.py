import argparse

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')

#==== ENV related arguments ====#
parser.add_argument("--env",
    type=str,
    help="gym environment to load",
    default='4-tiles-ad-rooms'
)
parser.add_argument("--seed",
    type=int,
    help="env seed generation, 1 seed 1 config",
    default=-1
)
parser.add_argument("--rooms_in_row",
    type=int,
    help="env number of rooms in row",
    default='3'
)
parser.add_argument("--rooms_in_col",
    type=int,
    help="env number of rooms in col",
    default='3'
)

#==== Model arguments ====#
parser.add_argument("--allo_config",
    type=str,
    help="path to the allocentric model to load as yaml",
    default='runs/GQN_V2_AD/v2_GQN_AD_conv7x7_bi/GQN.yml'
)    
parser.add_argument('--memory_config',
    type=str,
    default='navigation_model/Services/memory_service/memory_graph_config.yml',
    help="path to the memory_graph_config as yaml",
)
parser.add_argument('--memory_load',
    default=None,
    help="enter the path to .map memory, None by default",
)

parser.add_argument('--lookahead',
    type=int,
    default=5,
    help="default lookahead the agent use to navigate",
)
#==== TESTS arguments ====#
parser.add_argument("--test",
    help="Test we want to run.\
    'key: manual keyboard entry, \
    'exploration': exploration ",
    default='key',
    type= str
)

parser.add_argument("--video",
    action='store_true',
    help="should we record the current test",
    default=False
)

parser.add_argument("--save_dir",
    type=str, 
    required=False,
    help="saving directory for videos or other files",
    default="tests_outputs"
)
