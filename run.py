#!/usr/bin/env python3
from control_eval.keyboard_control_navigation import MinigridInteraction as keyboard_interaction
from control_eval.automatic_env_run import run_test
from control_eval.arguments import parser


if __name__ == "__main__":
# instantiating the decorator
    
    flags = parser.parse_args()
    print('in runs flags', flags)

    if 'key' in flags.test :
        redraw_window = True
    else:
        redraw_window = False
    minigrid_interaction = keyboard_interaction(flags, redraw_window=redraw_window)
    if flags.test != 'key':
        # env_def= env_definition(flags)
        run_test(minigrid_interaction, flags)

   
        
  