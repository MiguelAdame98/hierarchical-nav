# Update 2022
# =============================================================================
# Ghent University 
# IDLAB of IMEC
# Daria de Tinguy - daria.detinguy at ugent.be
# =============================================================================

# Original Source
# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
from operator import itemgetter
from .modules import *
from sys import maxsize

class Experience(object):
    '''A single experience.

    An Experience object is used to point out a single point in the experience
    map, Thus, it must store its position in the map and activation of pose and
    view cell modules.
    '''
    _ID = 0
    _ghost_ID = 0

    def __init__(self, x_pc, y_pc, th_pc, x_m, y_m, facing_rad, view_cell, local_pose, ghost_exp):
        '''Initializes the Experience.

        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :param x_m: the position of axis x in the experience map.
        :param y_m: the position of axis x in the experience map.
        :param facing_rad: the orientation of the experience, in radians.
        :param view_cell: the last most activated view cell.
        :param ghost_exp: wether this experience has a view_cell or not (verified or expected exp)
        '''
        
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.x_m = x_m
        self.y_m = y_m
        self.facing_rad = facing_rad
        self.view_cell = view_cell
        self.links = []
        self.init_local_position = local_pose

        self.ghost_exp = ghost_exp
        if self.ghost_exp:
            self.id = Experience._ghost_ID
            Experience._ghost_ID += 1
        else:
            self.id = Experience._ID
            Experience._ID += 1



    def link_to(self, target, accum_delta_x, accum_delta_y,
                accum_delta_facing, active_link):
        '''Creates a link between this experience and a taget one.

        :param target: the target Experience object.
        :param accum_delta_x: accumulator in axis x computed by experience map.
        :param accum_delta_y: accumulator in axis y computed by experience map.
        :param accum_delta_facing: accumulator of orientation, computed by 
                                   experience map.
        '''

        print('np.arctan2(accum_delta_y, accum_delta_x)', accum_delta_y, accum_delta_x, np.arctan2(accum_delta_y, accum_delta_x) )
        print('self.facing_rad, accum_delta_facing', self.facing_rad, accum_delta_facing)
        d = np.sqrt(accum_delta_x**2 + accum_delta_y**2)
        heading_rad = signed_delta_rad(
            self.facing_rad,
            np.arctan2(accum_delta_y, accum_delta_x)
        )
        facing_rad = signed_delta_rad(
            self.facing_rad,
            accum_delta_facing
        )
        link = ExperienceLink(self, target, facing_rad, d, heading_rad, active_link)
        self.links.append(link)
        print('     _     ')
    def __repr__(self):
        # return "[ {} {} {} {} ]".format(self.id, self.x_m, self.y_m,
        # self.facing_rad)
        return "[{}]".format(self.id)

    def update_link(self,link, e1):
        #must change link both way 
        if link.active_link == True:
            delta_x = e1.x_m - self.x_m
            delta_y = e1.y_m - self.y_m
        else:
            delta_x = self.x_m - e1.x_m
            delta_y = self.y_m - e1.y_m 
        delta_facing = signed_delta_rad(self.facing_rad, e1.facing_rad)
        print('in update link',self.facing_rad, e1.facing_rad, delta_facing)

        
        link.d = np.sqrt(delta_x**2 + delta_y**2)
        link.heading_rad = signed_delta_rad(
            self.facing_rad,
            np.arctan2(delta_y, delta_x)
        )
        link.facing_rad = signed_delta_rad(
            self.facing_rad,
            delta_facing
        )
        return link

class ExperienceLink(object):
    '''A representation of connection between experiences.'''

    def __init__(self, parent, target, facing_rad, d, heading_rad, active_link):
        '''Initializes the link.

        :param parent: the Experience object that owns this link.
        :param target: the target Experience object.
        :param facing_rad: the angle (in radians) in the map context.
        :param d: the euclidean distance between the Experiences.
        :param heading_rad: the angle (in radians) between the Experiences.
        :param active_link: wether we use this link for dist correction or it's an inverse direction link used to have a bidirectional connection
        '''
        self.parent = parent
        self.target = target
        self.facing_rad = facing_rad
        self.d = d
        self.heading_rad = heading_rad
        self.active_link = active_link


class ExperienceMap(object):
    '''Experience Map module.'''

    def __init__(self, dim_xy=61, dim_th=36,
                 delta_exp_threshold=1.0, delta_pc_threshold= 1.0, correction=0.5, loops=100,
                 constant_adjust=False, **kwargs):
        '''Initializes the experience map.'''

        self.DIM_XY = dim_xy
        self.DIM_TH = dim_th

        self.DELTA_EXP_THRESHOLD = delta_exp_threshold
        self.DELTA_PC_THRESHOLD = delta_pc_threshold
        self.CORRECTION = correction
        self.LOOPS = loops
        self.constant_adjust = constant_adjust

        self.size = 0
        self.exps = []
        self.ghost_exps = []

        self.current_exp = None
        self.current_view_cell = None

        self.accum_delta_x = 0
        self.accum_delta_y = 0
        self.accum_delta_facing = np.pi / 2

        
    def _create_exp(self, x_pc, y_pc, th_pc, view_cell, local_pose):
        '''Creates a new Experience object.

        This method creates a new experience object, which will be a point the
        map.

        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :param view_cell: the last most activated view cell.
        :return: the new Experience object.
        '''
        self.size += 1
        x_m = self.accum_delta_x
        y_m = self.accum_delta_y
        facing_rad = clip_rad_180(self.accum_delta_facing)

        if self.current_exp is not None:
            x_m += self.current_exp.x_m
            y_m += self.current_exp.y_m
        print('in exp create local init_location', local_pose )
        exp = Experience(x_pc, y_pc, th_pc, x_m, y_m, facing_rad, view_cell, local_pose, ghost_exp=False)

        if self.current_exp is not None:
            #connect from prev to newly created exp
            self.current_exp.link_to(
                exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing, active_link=True)
            #and vice versa
            exp.link_to(
                self.current_exp, - self.accum_delta_x, - self.accum_delta_y, self.current_exp.facing_rad, active_link=False)
            try:
                print('linked exp', self.current_exp.id,' to ', self.current_exp.links[-1].target.id, 'f,h,d:', self.current_exp.links[-1].facing_rad, self.current_exp.links[-1].heading_rad, self.current_exp.links[-1].d)
                print('inverse link exp', exp.id,' to ', exp.links[-1].target.id, 'f,h,d:', exp.links[-1].facing_rad, exp.links[-1].heading_rad, exp.links[-1].d)
            except AttributeError:
                pass
        self.exps.append(exp)
        if view_cell is not None:
            view_cell.exps.append(exp)
        

        return exp

    def _create_ghost_exp(self, x_m, y_m, facing_rad, linked_exp = None):
        '''Creates a new Experience object.

        This method creates a new experience object, which will be a point the
        map.

        :param xm: global position x estimation in map
        :param ym: global position y estimation in map
        :param facing_rad: global facing estimation in map
        :return: the new Experience object.
        '''
        self.size += 0


        exp = Experience(x_pc=0, y_pc=0, th_pc=0, x_m= x_m, y_m = y_m, facing_rad = facing_rad, view_cell=None, ghost_exp=True)

        if linked_exp is not None:
            accum_delta_x = linked_exp.x_m - x_m
            accum_delta_y = linked_exp.y_m - y_m
            accum_delta_facing = signed_delta_rad(linked_exp.facing_rad ,facing_rad)
            print('In ghost exp creation accum_x, y and facing',exp.id, accum_delta_x,accum_delta_y,accum_delta_facing )
            print('Given real linked exp', [linked_exp.x_m, linked_exp.y_m,linked_exp.facing_rad], ' and ghost exp ', [exp.x_m, exp.y_m, exp.facing_rad])
            linked_exp.link_to(
                exp, accum_delta_x, accum_delta_y, accum_delta_facing, active_link=True)
            exp.link_to(
                linked_exp, accum_delta_x, accum_delta_y, accum_delta_facing, active_link=False)

        self.ghost_exps.append(exp)

        return exp

    def update_exp_wt_view_cell(self,updated_exp,  x_pc, y_pc, th_pc, new_view_cell= None, local_pose= None):
        view_cell_exps = updated_exp.view_cell.exps
    

        x_m = self.accum_delta_x
        y_m = self.accum_delta_y
        facing_rad = clip_rad_180(self.accum_delta_facing)

        x_m += updated_exp.x_m
        y_m += updated_exp.y_m

        updated_exp.x_pc = x_pc
        updated_exp.y_pc = y_pc
        updated_exp.th_pc = th_pc
        updated_exp.x_m = x_m
        updated_exp.y_m = y_m
        updated_exp.facing_rad = facing_rad

        if local_pose is not None:
            updated_exp.init_local_position = local_pose

        

        if new_view_cell is not None:
            for exp in view_cell_exps:
                if exp.id == updated_exp.id:
                    view_cell_exps.remove(exp)
                    break
            updated_exp.view_cell = new_view_cell
            new_view_cell.exps.append(updated_exp)
        
    def get_exp(self, exp_id):
        '''
        Experience lookup by id - assumes ids are monotonically rising
        '''
        index = exp_id
        if len(self.exps) >= index:
            index = len(self.exps) - 1

        while self.exps[index].id != exp_id:
            if self.exps[index].id > exp_id:
                index -= 1
            else:
                index += 1
                if index == len(self.exps):
                    index = 0

        return self.exps[index]

    def get_ghost_exp(self, ghost_exp_id):
        '''
        Ghost Experience lookup by id 
        '''

        index = len(self.ghost_exps) - 1

        while self.ghost_exps[index].id != ghost_exp_id:
            if self.ghost_exps[index].id > ghost_exp_id:
                index -= 1
            else:
                index += 1
                if index == len(self.ghost_exps):
                    index = 0
        return self.ghost_exps[index]

    def get_current_exp_id(self):
        ''' where are we?'''
        if self.current_exp is not None:
            return self.current_exp.id
        else:
            return -1

    def get_current_exp_view_cell_id(self):
        ''' where are we?'''
        if self.current_exp is not None:
            return self.current_exp.view_cell.id
        else:
            return -1

    def get_exp_view_cell_content(self,exp_id:int):
        if self.current_exp is not None and exp_id>=0:
            exp = self.get_exp(exp_id)
            return exp.view_cell
        else:
            return None

    def get_exps(self, wt_links = False):
        ''' get all exp in a dict'''
        map_experiences = []
        for exp in self.exps:
            experience = {}
            experience['id'] = exp.id
            experience['x'] = exp.x_m
            experience['y'] = exp.y_m
            experience['facing'] = exp.facing_rad
            experience['init_local_position'] = exp.init_local_position
            experience['observation'] = exp.view_cell.template
            experience['observation_entry_poses'] = exp.view_cell.relevant_poses
            #experience['ob_info'] = exp.view_cell.template_info
            if wt_links == True:
                experience['links'] = exp.links

            map_experiences.append(experience)
        return map_experiences

    def get_exp_as_dict(self, id=None):
        experience = {}
        if isinstance(id, int):
            exp = self.get(id)
        else:
            exp = self.current_exp
        if exp is not None:
            experience['id'] = exp.id
            experience['x'] = exp.x_m
            experience['y'] = exp.y_m
            experience['facing'] = exp.facing_rad
            experience['observation'] = exp.view_cell.template
            #experience['ob_info'] = exp.view_cell.template_info
            #print('exp ' + str(self.current_exp.id) +' map position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) +' facing: ' +str(self.current_exp.facing_rad))
        return experience

    def get_delta_exp(self, x,y, delta_x, delta_y):
        ''' return euclidian distance between two points'''
        if self.current_exp is None:
            delta_exp = np.inf
        else:
            delta_exp = euclidian_distance([x,y], [delta_x, delta_y]) 
            
            # print('accumulated dist from prev exp: ' + str(accum_delta_x) + ' y: '+str(accum_delta_y) )
            # print('current view position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) ) 
            # print('delta_exp and th', delta_exp , self.DELTA_EXP_THRESHOLD )
        return delta_exp   

    def accumulated_delta_location(self, vtrans, vrot):
        #% integrate the delta x, y, facing
        accum_delta_facing = clip_rad_180(self.accum_delta_facing + vrot)
        accum_delta_x = self.accum_delta_x + vtrans * np.cos(self.accum_delta_facing)
        accum_delta_y = self.accum_delta_y + vtrans * np.sin(self.accum_delta_facing)

        return accum_delta_facing, accum_delta_x, accum_delta_y

    def delta_exp_above_thresold(self, delta_exp:float)->bool:
        print('delta exp and delta threshold', delta_exp, self.DELTA_EXP_THRESHOLD)
        return delta_exp > self.DELTA_EXP_THRESHOLD
    
    def delta_pc_above_thresold(self, delta_pc):
        print('delta exp and delta threshold', delta_pc, self.DELTA_PC_THRESHOLD)
        return delta_pc > self.DELTA_PC_THRESHOLD

    def get_global_position(self):
        if self.current_exp is not None:
            current_x_m =  self.accum_delta_x + self.current_exp.x_m
            current_y_m =  self.accum_delta_y + self.current_exp.y_m
        else:
            current_x_m =  self.accum_delta_x
            current_y_m =  self.accum_delta_y
        return [current_x_m, current_y_m, self.accum_delta_facing]
    
    def get_exp_global_position(self, exp:object=-1)->list:
        if isinstance(exp, int):
            if self.current_exp is not None:
                return [self.current_exp.x_m, self.current_exp.y_m, self.current_exp.facing_rad] 
            else:
                raise "get_exp_global_position can't accept element" + str(exp) +'of type' + str(type(exp))
        elif exp is None:
            raise "get_exp_global_position can't accept element" + str(exp) +'of type' + str(type(exp))
        return [exp.x_m, exp.y_m, exp.facing_rad] 
        
            

    def __call__(self, view_cell, vtrans, vrot, x_pc, y_pc, th_pc, adjust, local_pose=None, view_cell_copy=None):
        '''Run an interaction of the experience map.

        :param view_cell: the last most activated view cell.
        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :param vtrans: the translation of the robot given by odometry.
        :param vrot: the rotation of the robot given by odometry.
        :param adjust: run the map adjustment procedure.
        '''

        self.accum_delta_facing,self.accum_delta_x, self.accum_delta_y  = self.accumulated_delta_location(vtrans,vrot)
        #delta_prev_exp = self.get_delta_exp(0,0,self.accum_delta_x, self.accum_delta_y)
        #print('accumulated dist from prev exp: ' + str(self.accum_delta_x) + ' y: '+str(self.accum_delta_y) )
        
        #delta_exp_above_thresold = self.delta_exp_above_thresold(delta_prev_exp)
        adjust_map = False

        if self.current_exp != None:
        
            #approximate curent position
            current_GP = self.get_global_position()
            
            print('CHECK CURRENT GP X,Y,TH', current_GP )
        
            delta_exps = []
        
            for e in self.exps:
                delta_exp = self.get_delta_exp(e.x_m,e.y_m, current_GP[0], current_GP[1])
                delta_exps.append(delta_exp)
            
            min_delta_GP_id = np.argmin(delta_exps)
            min_delta_GP_val = delta_exps[min_delta_GP_id]
            #print('delta_exps',delta_exps)
            delta_pc = np.sqrt(
                min_delta(self.current_exp.x_pc, x_pc, self.DIM_XY)**2 +
                min_delta(self.current_exp.y_pc, y_pc, self.DIM_XY)**2 
                #+ min_delta(self.current_exp.th_pc, th_pc, self.DIM_TH)**2
                )
            print('current experience x: ',self.current_exp.x_m, ', y:',self.current_exp.y_m,' and th:', self.current_exp.facing_rad)
            print('current position x_pc: ',x_pc, ', y_pc:',y_pc,' and th_pc:', th_pc)
            print('delta_pc between exp and curreny pc',delta_pc)

            close_exp_link_exists = False
            if min_delta_GP_id != self.current_exp.id:
                for linked_exp in [l.target for l in self.current_exp.links]:
                    if linked_exp.id == min_delta_GP_id:
                        close_exp_link_exists = True
                        break
            else:
               #If the closest exp is the current exp, then a link between the two does exist
               close_exp_link_exists = True 
        else:
            min_delta_GP_val = np.inf
            min_delta_GP_id= 0
            delta_pc = np.inf

        #print('accumulated dist in exp map', self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing,'is delta_exp_above_thresold?', delta_prev_exp, 'closest exp dist', min_delta_GP_val, min_delta_GP_id)
        
        # if current exp is None, just select first matching?
        if self.current_exp is None and Experience._ID == 0 :
            print('first experience created')
            exp = self._create_exp(x_pc, y_pc, th_pc, view_cell, local_pose)

            self.current_exp = exp
            self.accum_delta_x = 0
            self.accum_delta_y = 0
            self.accum_delta_facing = self.current_exp.facing_rad
            #self.current_exp = view_cell.exps[0]
        #if we loaded a memory map, then we need to get experience matching view cell
        elif self.current_exp is None:
            self.current_exp = view_cell.exps[0] #NOTE: this works considering that 1 exp has 1 ob
            print('we are initialising position to',self.current_exp.id, 'extracted from memory, based on observation', view_cell.id)

            self.accum_delta_x = 0
            self.accum_delta_y = 0
            self.accum_delta_facing = self.current_exp.facing_rad
                   

        #We have a new view but it's close to a previous experience
        elif len(view_cell.exps) == 0 and min_delta_GP_val < self.DELTA_EXP_THRESHOLD:
            print('too close from exp ', min_delta_GP_id,', dist between exp and here', min_delta_GP_val)
            # exp = self.get_exp(min_delta_GP_id)
            # delta_exp_pc = np.sqrt(
            #     min_delta(self.current_exp.x_pc, exp.x_pc, self.DIM_XY)**2 +
            #     min_delta(self.current_exp.y_pc, exp.y_pc, self.DIM_XY)**2 
            #     + min_delta(self.current_exp.th_pc, exp.th_pc, self.DIM_TH)**2
            #     )
            # print('checking motion between two considered experiences', delta_exp_pc)
            if min_delta_GP_id != self.current_exp.id: #delta_exp_pc > self.DELTA_PC_THRESHOLD: #if we are not considering same exp
                print('we are close looping with exp', min_delta_GP_id,' discarding newly generated view_cell ',view_cell.id)
                adjust_map = True
                close_loop_exp = self.exps[min_delta_GP_id]
                # see if the exp near by already has a link to the current exp
                link_exists = False
                for linked_exp in [l.target for l in self.current_exp.links]:
                    if linked_exp == close_loop_exp:
                        link_exists = True

                if not link_exists:
                    self.current_exp.link_to(
                        close_loop_exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing, active_link=True)
                    close_loop_exp.link_to(
                        self.current_exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing, active_link=False)
                    
                self.current_exp = close_loop_exp
                self.accum_delta_x = 0
                self.accum_delta_y = 0
                print('We keep current GP facing rad, this might be an issue in real environment')
                #self.accum_delta_facing = self.current_exp.facing_rad
                #print('new current exp id and its view cell id:', self.current_exp.id, self.current_exp.view_cell.id)
            else:
                print('replacing previously generated view_cell', self.current_exp.view_cell.id,' of exp ',self.current_exp.id)
                print('OLD current exp position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) , ' facing: '+str(self.current_exp.facing_rad) \
                      , ' local pose: '+str(self.current_exp.init_local_position) ) 
                self.update_exp_wt_view_cell(self.current_exp, x_pc, y_pc, th_pc, view_cell, local_pose)
                print('NEW current exp position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) , ' facing: '+str(self.current_exp.facing_rad) \
                      , ' local pose: '+str(self.current_exp.init_local_position)) 
                

                for link in self.current_exp.links:
                    e1 = link.target
                    print('original link of current exp', self.current_exp.id, ' to ', e1.id, 'f,h,d:', link.facing_rad, link.heading_rad, link.d)
                    #change link from source to linked exp
                    self.current_exp.update_link(link, e1)
                    print('updated link of current exp', self.current_exp.id, ' to ', e1.id, 'f,h,d:', link.facing_rad, link.heading_rad, link.d)
                    
                    #change link from linked exp to source
                    for l in e1.links:
                        if l.target == self.current_exp:
                            print()
                            print('original link of linked exp', e1.id,' to ', l.target.id, 'f,h,d:', l.facing_rad, l.heading_rad, l.d)
                            e1.update_link(l, self.current_exp)
                            print('updated link of linked exp', e1.id, ' to ', l.target.id, 'f,h,d:', l.facing_rad, l.heading_rad, l.d)
                            break

                self.accum_delta_x = 0
                self.accum_delta_y = 0
                self.accum_delta_facing = self.current_exp.facing_rad

        # if the vt is new AND the global pose x,y,th is far enough from any prev experience create a new experience
        elif len(view_cell.exps) == 0 :
            #if current location is far enough from prev one, else, view cells are considered as in conflict
            print('no exp in view, len =' ,len(view_cell.exps), 'closest exp dist', min_delta_GP_val )
            exp = self._create_exp(x_pc, y_pc, th_pc, view_cell,local_pose)

                            
            self.current_exp = exp
            self.accum_delta_x = 0
            self.accum_delta_y = 0
            self.accum_delta_facing = self.current_exp.facing_rad
            
        

        # if the vt has changed (but isn't new) search for the matching exp
        elif view_cell != self.current_exp.view_cell:
            # find the exp associated with the current vt and that is under the
            # threshold distance to the centre of pose cell activity
            # if multiple exps are under the threshold then don't match (to reduce
            # hash collisions)
            print('new selected view_cell is different from previous one but has exp (known view)')
            print('view_cell ID and current exp view cell id', view_cell.id, self.current_exp.view_cell.id)
            adjust_map = True
            matched_exp = None

            delta_view_exps = []
            n_candidate_matches = 0
   
            for (i, e) in enumerate(view_cell.exps):
                #print('exp ' + str(i) +' view position x: ' + str(e.x_m) + ' y: '+str(e.y_m) )
                
                delta_view_exp = self.get_delta_exp(e.x_m,e.y_m,current_GP[0], current_GP[1])
                delta_view_exps.append(delta_view_exp)

                if delta_view_exp < self.DELTA_EXP_THRESHOLD:
                    n_candidate_matches += 1

            if n_candidate_matches > 1:
                print('more than 1 exp could correspond, candidate match > 1', n_candidate_matches)
                pass

            else:
                print('delta_view_exps of all exp of this view', delta_view_exps)
                
                min_delta_GP_id = np.argmin(delta_view_exps)
                min_delta_GP_val = delta_view_exps[min_delta_GP_id]
                #NOTE: static 2* added that won't be pertinent for all environments
                if min_delta_GP_val < 2 * self.DELTA_EXP_THRESHOLD :
                    print('the delta between exp' , view_cell.exps[min_delta_GP_id].id ,' and ' ,self.current_exp.id, ' allow for a close-loop')
                    matched_exp = view_cell.exps[min_delta_GP_id]

                    # see if the prev exp already has a link to the current exp
                    link_exists = False
                    for linked_exp in [l.target for l in self.current_exp.links]:
                        if linked_exp == matched_exp:
                            link_exists = True
                            break

                    if not link_exists:
                        self.current_exp.link_to(
                            matched_exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing, active_link=True)
                        matched_exp.link_to(
                            self.current_exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing, active_link=False)
                
                if matched_exp is None:
                    print(
                        "warning, no matched experience although matching view cell ", min_delta_GP_val)
                    #View_cell_copy contains the same ob as view_cell but without experience attached and with a new ID
                    #TODO: add view_cell_copy instead of view_cell
                    matched_exp = self._create_exp(
                        x_pc, y_pc, th_pc, view_cell_copy,local_pose)

                self.current_exp = matched_exp
                self.accum_delta_x = 0
                self.accum_delta_y = 0
                #commented line below as we don't narrow close loop around position but place, this means no strict position correction in odom
                #self.accum_delta_facing = self.current_exp.facing_rad
        # print('current experience id and matching view',self.current_exp.id, view_cell.id)
        # print('exp ' + str(self.current_exp.id) +' map position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) +' facing: ' +str(self.current_exp.facing_rad))
        
        #If nothing particular and we are close from a previous experience, we link them if not already
        elif min_delta_GP_val < self.DELTA_EXP_THRESHOLD and close_exp_link_exists != True : 
            # if the closest exp has no link to the current exp
            print('linking current exp ', self.current_exp.id, ' to closest exp ', min_delta_GP_id)
            min_delta_GP_exp = self.get_exp(min_delta_GP_id)

            delta_x = min_delta_GP_exp.x_m - self.current_exp.x_m
            delta_y = min_delta_GP_exp.y_m - self.current_exp.y_m
            self.current_exp.link_to(
                min_delta_GP_exp, delta_x, delta_y, self.current_exp.facing_rad, active_link=True)
            min_delta_GP_exp.link_to(
                self.current_exp, delta_x, delta_y, self.current_exp.facing_rad, active_link=False)

        #if we have ghost nodes, we check if we are moving near one
        elif len(self.ghost_exps) > 0 :
            delta_ghost_exps = []
            for e in self.ghost_exps:
                delta_ghost_exp = self.get_delta_exp(e.x_m,e.y_m,current_GP[0], current_GP[1])
                delta_ghost_exps.append([delta_ghost_exp, e.id])
            print('delta_ghost_exps', delta_ghost_exps)
            closest_ghost= sorted(delta_ghost_exps, key=itemgetter(0))[0]
            min_ghost_delta_val = closest_ghost[0]
            min_ghost_delta_id = closest_ghost[1]
            
            print('closest ghost node dist and id', min_ghost_delta_val, min_ghost_delta_id )

            if min_ghost_delta_val <= self.DELTA_EXP_THRESHOLD:
                ghost_exp = self.get_ghost_exp(min_ghost_delta_id)
                self.link_exp_through_ghost(self.current_exp, ghost_exp)
                self.ghost_exps.remove(ghost_exp)

        #If we replaced the view cell in same exp, we need to update the global position to match init_local_position
        if self.current_exp.view_cell.to_update:
            print('Updating global location of the exp ', self.current_exp.id , 'to match the new init_local_position of view cell:',local_pose )

            print('OLD current exp position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) , ' facing: '+str(self.current_exp.facing_rad) \
                      , ' local pose: '+str(self.current_exp.init_local_position), 'view cell id', self.current_exp.view_cell.id ) 
            self.update_exp_wt_view_cell(self.current_exp, x_pc, y_pc, th_pc, new_view_cell= None, local_pose= local_pose)
            print('NEW current exp position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) , ' facing: '+str(self.current_exp.facing_rad) \
                , ' local pose: '+str(self.current_exp.init_local_position), 'view cell id', self.current_exp.view_cell.id) 
            self.current_exp.view_cell.to_update = False
            for link in self.current_exp.links:
                e1 = link.target
                print('original link of current exp', self.current_exp.id, link.facing_rad, link.heading_rad, link.d)
                #change link from source to linked exp
                self.current_exp.update_link(link, e1)
                print('updated link of current exp', self.current_exp.id, link.facing_rad, link.heading_rad, link.d)
                
                #change link from linked exp to source
                for l in e1.links:
                    if l.target == self.current_exp:
                        print('original link of linked exp', e1.id, l.facing_rad, l.heading_rad, l.d)
                        e1.update_link(l, self.current_exp)
                        print('updated link of linked exp', e1.id, l.facing_rad, l.heading_rad, l.d)
                        break
        
      
        print('self.current_exp.init_local_position',  self.current_exp.init_local_position)

        #print('adjust map?', self.constant_adjust, adjust, adjust_map)
        if not self.constant_adjust:
            if not adjust or not adjust_map:
                return

        
        # Iteratively update the experience map with the new information
        for i in range(0, self.LOOPS):
            for e0 in self.exps:
                for l in e0.links:
                    print(e0.id, 'is linked to', l.target.id)
                    # e0 is the experience under consideration
                    # e1 is an experience linked from e0
                    # l is the ACTIVE link object which contains additional heading
                    # info
                    if l.active_link == True:
                        e1 = l.target
                        print('BEFORE CORRECTION IN LOOP exp ' + str(e0.id) +' map position x: ' + str(e0.x_m) + ' y: '+str(e0.y_m) +' facing: ' +str(e0.facing_rad) \
                            + ' Ghost?' +str(e0.ghost_exp) )
                        print('BEFORE CORRECTION IN LOOP exp ' + str(e1.id) +' map position x: ' + str(e1.x_m) + ' y: '+str(e1.y_m) +' facing: ' +str(e1.facing_rad) \
                            + ' Ghost?' +str(e1.ghost_exp) )
                        # correction factor
                        cf = self.CORRECTION

                        # work out where exp0 thinks exp1 (x,y) should be based on
                        # the stored link information
                        lx = e0.x_m + l.d * np.cos(e0.facing_rad + l.heading_rad)
                        ly = e0.y_m + l.d * np.sin(e0.facing_rad + l.heading_rad)

                        # determine the angle between where e0 thinks e1's facing
                        # should be based on the link information
                        df = signed_delta_rad(e0.facing_rad + l.facing_rad,
                                            e1.facing_rad)
                        print('df and link facing rad', df, l.facing_rad)
                        # correct e0 and e1 (x,y) by equal but opposite amounts
                        # a 0.5 correction parameter means that e0 and e1 will be
                        # fully corrected based on e0's link information
                        #+
                        ## correct e0 and e1 facing by equal but opposite amounts
                        # a 0.5 correction parameter means that e0 and e1 will be
                        # fully corrected based on e0's link information
                        #+
                        # If one of the exp is a ghost exp, then the ghost should not influence the real exp
                        # #NOTE: IN OUR CONTEXT, updating facing doesn't correct any shift
                        if e1.ghost_exp != True:
                            print('check e0 correction', cf, lx, ly)
                            e0.x_m = e0.x_m + (e1.x_m - lx) * cf
                            e0.y_m = e0.y_m + (e1.y_m - ly) * cf
                            #e0.facing_rad = clip_rad_180(e0.facing_rad + df * cf)
                        if e0.ghost_exp != True:
                            print('check e1 correction', cf, lx, ly)
                            e1.x_m = e1.x_m - (e1.x_m - lx) * cf
                            e1.y_m = e1.y_m - (e1.y_m - ly) * cf
                            #e1.facing_rad = clip_rad_180(e1.facing_rad - df * cf)

                        print('AFTER CORRECTION IN LOOP exp ' + str(e0.id) +' map position x: ' + str(e0.x_m) + ' y: '+str(e0.y_m) +' facing: ' +str(e0.facing_rad))
                        print('AFTER CORRECTION IN LOOP exp ' + str(e1.id) +' map position x: ' + str(e1.x_m) + ' y: '+str(e1.y_m) +' facing: ' +str(e1.facing_rad))   
                    print() 
        for e0 in self.exps:           
            print('aFTER CORRECTIONS  exp ' + str(e0.id) +' map position x: ' + str(e0.x_m) + ' y: '+str(e0.y_m) +' facing: ' +str(e0.facing_rad))
          

        #print('AFTER CORRECTION  exp ' + str(self.current_exp.id) +' map position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) +' facing: ' +str(self.current_exp.facing_rad))
        # print( )
        # print('____')
        
        return


    def create_ghost_exps(self,origin_exp:object, ghost_poses:list):
        ''' Params:
        exp_id: the experience the ghost is linked to
        ghost_poses: GP of the ghost nodes to create. -
        We create ghost nodes for all poses then check if 
        any ghost node is near an experience to operate close looping
        '''

        #for each potential ghost pose
        for GP_pose in ghost_poses:
            self._create_ghost_exp(GP_pose[0], GP_pose[1], GP_pose[2], linked_exp=origin_exp)
            
        #--- we clean ghost node and close loop when possible
        self.sort_ghost_exps()

    def sort_ghost_exps(self):
        """
        Check all exps and all ghost nodes and close loop/ erase ghost nodes 
        when a ghost node is near an experience
        """
        for exp in self.exps: 
            print('exp ', exp.id, [exp.x_m, exp.y_m, exp.facing_rad])   
            exp_GP = [exp.x_m, exp.y_m, exp.facing_rad]
            for ghost_exp in self.ghost_exps[:]:
                ghost_GP = [ghost_exp.x_m, ghost_exp.y_m, ghost_exp.facing_rad]
                print('ghost exp ', ghost_exp.id, [ghost_exp.x_m, ghost_exp.y_m, ghost_exp.facing_rad])   
                delta_exp = euclidian_distance(ghost_GP, exp_GP)
                print('delta dist', delta_exp,  self.DELTA_EXP_THRESHOLD)
                if self.delta_exp_above_thresold(delta_exp):
                    self.link_exp_through_ghost(exp, ghost_exp)
                    print('remove ghost exp as close to exp by dist = ', delta_exp)
                self.ghost_exps.remove(ghost_exp)


    def link_exp_through_ghost(self, exp:object, ghost_exp:object):
        ''' 
        Linking the ghost node origin exp to the exp measured as nearby 
        and erasing ghost node
        '''
        exp_links = exp.links
        exp_links_ids = []
        for l in exp_links:
            exp_links_ids.append(l.target.id)
        for link in ghost_exp.links:
            ghost_target_id = link.target.id
            #If the experience do not posses a link the ghost at the same position does, then we add it
            if ghost_target_id not in exp_links_ids and ghost_target_id != exp.id :
                linked_exp = self.get_exp(ghost_target_id)
                accum_delta_x = linked_exp.x_m - exp.x_m
                accum_delta_y = linked_exp.y_m - exp.y_m
                accum_delta_facing = signed_delta_rad(linked_exp.facing_rad ,exp.facing_rad)
                print()
                print('In ghost exp close looping exp id and linked exp id', exp, linked_exp.id)
                print(' x y th of exp and linked exp',  [exp.x_m,exp.y_m, exp.facing_rad], [linked_exp.x_m, linked_exp.y_m, linked_exp.facing_rad])
                print(' accum_x, y and facing',accum_delta_x,accum_delta_y,accum_delta_facing )
                exp.link_to( 
                linked_exp,accum_delta_x,accum_delta_y,accum_delta_facing, active_link=True)
                
                linked_exp.link_to(
                exp,accum_delta_x,accum_delta_y,accum_delta_facing, active_link=False)
