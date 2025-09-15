from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np
import random 
from collections import deque

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class AisleDoorRooms(MiniGridEnv):
    """
    colored rooms connected by x tiles corridors
    """

    def __init__(
        self,
        size= 24,
        agent_start_pos= None,
        agent_start_dir= 0,
        rooms_size = 4,
        max_steps= None,
        rooms_in_row = 3,
        rooms_in_col = 3,
        corridor_length =5,
        automatic_door = True,
        wT_around = 6,
        wT_size = 1,
        obstacle_rate=0.15,          # Add this
        debug=True

    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.automatic_door = automatic_door
        self.wT_around = wT_around
        self.wT_size= wT_size
        self.obstacle_rate = obstacle_rate
        self.debug = debug
        self.obstacles_spawned = False  
        self.current_obstacles = set()
        self.color_idx = {
            'red' : 0,
            'green' : 1,
            'blue'  : 2,
            'purple': 3,
            'white' : 4,   # keep for doors/goals
        }
        # Prefer not to paint rooms 'white' since doors/goals are white
        self._room_palette = [c for c in self.color_idx.keys() if c != 'white']


        self.rooms_size = rooms_size
        
        if isinstance(corridor_length, int) and corridor_length>0:
            self.corridor_length = corridor_length
            self.random_corridor_length = False
        else:
            self.random_corridor_length = True

        if self.random_corridor_length:
            self.corridor_length = self._rand_int(3,7)

        self.rooms_in_row = rooms_in_row
        self.rooms_in_col = rooms_in_col        

        width = (self.rooms_size * rooms_in_col + (self.corridor_length+1) * (rooms_in_col-1)-1  )
        height = (self.rooms_size * rooms_in_row + (self.corridor_length+1) * (rooms_in_row-1)-1 )
        #print('width, height',width, height)
        if width <= 20 :
            width+=1
        if height <= 20 :
            height+=1

        if width >= 30 :
            width-=1
        if height >= 30 :
            height-=1
        # print(width, height)
        if width >= 40 :
            width-=1
        if height >= 40 :
            height-=1
        
        # self.static_room = static_room
        # self.closed = closed
        self.color_idx = {
            'red' : 0,
            'green' : 1,
            'blue'  : 2,
            'purple': 3,
            'white' : 4,
            
        }

        super().__init__(
            grid_size=None,
            width=width,
            height=height,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=False,
            
            
        )     

    
    def _gen_grid(self, width, height):
        print("1TFFFFFFFFF")       
        rooms_in_row = self.rooms_in_row
        rooms_in_col = self.rooms_in_col
        room_w = self.rooms_size
        room_h = self.rooms_size

        # >>> PATCH 1: initialize tracking
        self._init_room_tracking(width, height)

        # Create an empty grid ...
        self.grid = Grid(width, height)
        self.grid.grid = [Floor('black')] * width * height

        # Surrounding walls ...
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        self.grid.wall_rect(0, 0, width, height)

        balls_rooms = []
        while len(balls_rooms) < self.wT_around:
            col_room = self._rand_int(0, rooms_in_col)
            row_room = self._rand_int(0, rooms_in_row)
            if [col_room, row_room] not in balls_rooms:
                balls_rooms.append([col_room, row_room])

        pos_vert_R = None
        pos_horz_B = [None] * rooms_in_col
        agent_pose_options = []

        # For each row of rooms
        for row_inc in range(0, rooms_in_row):
            pos_vert_R = None
            for col_inc in range(0, rooms_in_col):
                xL = col_inc * (room_w + self.corridor_length)
                yT = row_inc * (room_h + self.corridor_length)
                xR = xL + room_w 
                yB = yT + room_h

                # choose color
                if hasattr(self, "_rand_elem"):
                    color = self._rand_elem(self._room_palette)
                else:
                    import random
                    color = random.choice(self._room_palette)
                # >>> PATCH 2: map room area + store meta (color, bounds)
                self._register_room_area(col_inc, row_inc, xL, yT, xR, yB, color)

                # fill interior with colored floor (your existing loop)
                for b in range(xL+1, xR):
                    for c in range(yT+1, yB):
                        self.put_obj(Floor(color), b, c)

                # ----- LEFT neighbor connection (upper wall/door in your terms) -----
                if col_inc > 0:
                    self.grid.vert_wall(xL, yT, room_h+1)
                    self.grid.set(xL, pos_vert_R[1], Floor('black'))

                    # door between left room and current corridor
                    door_x = xL - int(self.corridor_length/2)
                    door_y = pos_vert_R[1]
                    self.put_obj(Door(color='black'), door_x, door_y)

                    self._register_door(door_x, door_y, roomA=(col_inc-1, row_inc), roomB=(col_inc, row_inc))
                    self._register_corridor_run(
                        range(max(0, xL - self.corridor_length), xL),  # full length, clamped at 0
                        [door_y]
                    )

                    corridor_depth = self._rand_int(0, 2)
                    agent_pose_options.append((door_x - corridor_depth, door_y, 0))

                # ----- RIGHT neighbor preparation (creates corridor to the right) -----
                if col_inc + 1 < rooms_in_col:
                    self.grid.vert_wall(xR, yT, room_h+1)
                    pos_vert_R = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos_vert_R, Floor('black'))

                    # walls above/below corridor; walkable stripe at y=pos_vert_R[1]
                    self.grid.horz_wall(xR, pos_vert_R[1]-1, self.corridor_length)  
                    self.grid.horz_wall(xR, pos_vert_R[1]+1, self.corridor_length)

                    # >>> PATCH 3b: record corridor run to the right
                    self._register_corridor_run(range(xR, xR + self.corridor_length), [pos_vert_R[1]])

                    corridor_depth = self._rand_int(0, 2)
                    agent_pose_options.append((xR + int(self.corridor_length/2) + corridor_depth, pos_vert_R[1], 2))

                # ----- TOP neighbor connection -----
                if row_inc > 0:
                    self.grid.horz_wall(xL, yT, room_w+1)
                    self.grid.set(pos_horz_B[col_inc][0], yT, Floor('black'))

                    door_x = pos_horz_B[col_inc][0]
                    door_y = yT - int(self.corridor_length/2)
                    self.put_obj(Door(color='black'), door_x, door_y)

                    # >>> PATCH 3c: record doorway + corridor stripe up
                    self._register_door(door_x, door_y, roomA=(col_inc, row_inc-1), roomB=(col_inc, row_inc))
                    self._register_corridor_run(
                        [door_x],
                        range(max(0, yT - self.corridor_length), yT)  # full length, clamped
                    )

                    corridor_depth = self._rand_int(0, 2)
                    agent_pose_options.append((door_x, door_y - corridor_depth, 1))

                # ----- BOTTOM neighbor preparation -----
                if row_inc + 1 < rooms_in_row:
                    self.grid.horz_wall(xL, yB, room_w+1)
                    pos_horz_B[col_inc] = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos_horz_B[col_inc], Floor('black'))

                    # corridor vertical walls; walkable stripe at x=pos_horz_B[col_inc][0]
                    self.grid.vert_wall(pos_horz_B[col_inc][0]-1, yB, self.corridor_length)  
                    self.grid.vert_wall(pos_horz_B[col_inc][0]+1, yB, self.corridor_length)

                    # >>> PATCH 3d: record corridor run downward
                    cx = pos_horz_B[col_inc][0]
                    self._register_corridor_run([cx], range(yB, yB + self.corridor_length))
                    

                    corridor_depth = self._rand_int(0, 2)
                    agent_pose_options.append((cx, yB + int(self.corridor_length/2) + corridor_depth, 3))

                # Goals in selected rooms (unchanged)
                """if [col_inc, row_inc] in balls_rooms:
                    if self.wT_size == 1:
                        x, y = self._rand_int(xL+1, xR), self._rand_int(yT+1, yB)
                        self.put_obj(Goal('white'), x, y)
                        self.goal = (color, x, y)
                    else:
                        x = self._rand_int(xL+1, xR-1)
                        y = self._rand_int(yT+1, yB-1)
                        self.put_obj(Goal('white'), x, y)
                        self.put_obj(Goal('white'), x+1, y)
                        self.put_obj(Goal('white'), x, y+1)
                        self.put_obj(Goal('white'), x+1, y+1)"""

        # Place the agent (unchanged)
        if self.agent_start_pos == None:
            index = self._rand_int(0, len(agent_pose_options))
            self.starting_agent_pos = np.array([agent_pose_options[index][0], agent_pose_options[index][1]])
            self.agent_start_dir = agent_pose_options[index][2]
            self.agent_pos = self.starting_agent_pos   
            self.agent_dir = self.agent_start_dir
        else:
            self.agent_pos = np.asarray(self.agent_start_pos)
            self.agent_dir = self.agent_start_dir

        # >>> PATCH 4: seed visit tracker with starting location
        self._last_room_id = self.get_room_id_at(self.agent_pos[0], self.agent_pos[1])
        if self._last_room_id is not None and self._last_room_id not in self._visited_rooms_set:
            self._visited_rooms_set.add(self._last_room_id)
            self.visited_rooms_order.append({
                'room': self._last_room_id,
                'color': self.get_room_color_by_room(self._last_room_id),
                'entered_at_step': 0,
                'entry_xy': (int(self.agent_pos[0]), int(self.agent_pos[1])),
            })
        
        self.vel_ob = [0,0]
        self.encoded_action = None      

        if self.grid.get(*self.agent_pos).type == 'door' :
            self.door_open = self.agent_pos
            self.grid.get(*self.agent_pos).toggle(self, self.agent_pos)
        elif self.grid.get(*self.front_pos).type == 'door':
            self.door_open = self.front_pos
            self.grid.get(*self.front_pos).toggle(self, self.front_pos)
        else:
            self.door_open = [0,0]

        
        #self.grid.set(14, 10, Wall)
        #self.grid.set(3,4,Wall)
        #self.grid.vert_wall(21, 11, 1)
        #self.grid.vert_wall(22, 11, 1)
        #self.grid.vert_wall(20, 11, 1)
        # self.put_obj(Goal(),self.agent_pos[0]+1, self.agent_pos[1])
        self.mission = "Motion in color distinct rooms environment"



    def step(self, action):
        obs, reward, done, info = super().step(action)

        fwd_cell = self.grid.get(*self.front_pos)
        if self.automatic_door == True and fwd_cell != None and fwd_cell.type == 'door':
            #if a door up front, open it
            obs_2,_,_,_= super().step(self.actions.toggle, automatic_door=self.automatic_door)
            obs['door_open'] = obs_2['door_open']
            obs['image'] = obs_2['image']


        if obs['vel_ob'][0] ==obs['vel_ob'][1] :
            real_action = [0,0,0]
        else:
            real_action = obs['action'].copy()

        self.current_pose = self.action_to_pose(real_action, self.current_pose) 
        try:
            self._mark_room_visit(self.agent_pos[0], self.agent_pos[1])
        except Exception:
            pass
        obs['pose'] = self.current_pose
        return obs, reward, done, info
    
    def _init_room_tracking(self, width: int, height: int):
        """Call at the very start of _gen_grid()."""
        # Per-room metadata
        self.room_meta = {}             # dict[(col,row)] -> {'bounds':(x1,y1,x2,y2), 'color':str, 'doorways':[(x,y),...]}
        # Reverse index: grid cell -> room id
        self.xy_to_room = {}            # dict[(x,y)] -> (col,row)
        # Corridor and door coordinates
        self.corridor_xy = set()        # set[(x,y)]
        self.door_xy = []               # list[(x,y)]
        # Visit tracking (discovery order)
        self.visited_rooms_order = []   # list[{'room':(col,row), 'color':str, 'entered_at_step':int, 'entry_xy':(x,y)}]
        self._visited_rooms_set = set()
        self._last_room_id = None

    def _register_room_area(self, col: int, row: int, xL: int, yT: int, xR: int, yB: int, color: str):
        """
        Record per-room metadata and map interior floor cells (exclude walls).
        xL,yT inclusive left/top; xR,yB exclusive right/bottom boundary used by your code.
        """
        # Interior floor (exclude the 1-cell walls you build around)
        x1, y1 = xL + 1, yT + 1
        x2, y2 = xR - 1, yB - 1
        self.room_meta[(col, row)] = {
            'bounds': (x1, y1, x2, y2),
            'color':  color,
            'doorways': []
        }
        for x in range(x1, xR):
            for y in range(y1, yB):
                self.xy_to_room[(x, y)] = (col, row)

    def _register_door(self, x: int, y: int, roomA: tuple[int,int] | None = None, roomB: tuple[int,int] | None = None):
        self.door_xy.append((x, y))
        # track discovered doorway cells; also mark as corridor for indexing
        self.corridor_xy.add((x, y))
        # Optional: attach doorway to adjacent rooms if provided
        if roomA in self.room_meta:
            self.room_meta[roomA]['doorways'].append((x, y))
        if roomB in self.room_meta:
            self.room_meta[roomB]['doorways'].append((x, y))

    def _register_corridor_run(self, xs: range | list[int], ys: range | list[int]):
        """Mark a corridor 'walkable stripe' we created between rooms."""
        for x in (xs if isinstance(xs, range) else list(xs)):
            for y in (ys if isinstance(ys, range) else list(ys)):
                self.corridor_xy.add((x, y))

    def _mark_room_visit(self, x: int, y: int):
        """
        Call after agent position is updated (each step). Uses self.step_count if available.
        Detects transitions and logs first-time visits.
        """
        rid = self.xy_to_room.get((int(x), int(y)))  # (col,row) or None (corridor)
        # Transition detection
        if rid != self._last_room_id:
            if rid is not None and rid not in self._visited_rooms_set:
                self._visited_rooms_set.add(rid)
                color = self.room_meta.get(rid, {}).get('color', 'gray')
                step_no = int(getattr(self, 'step_count', 0))
                self.visited_rooms_order.append({
                    'room': rid,
                    'color': color,
                    'entered_at_step': step_no,
                    'entry_xy': (int(x), int(y)),
                })
        self._last_room_id = rid

    # --- Convenience getters (useful for plotting and EMAP color bridging) -----

    def get_room_id_at(self, x: int, y: int):
        return self.xy_to_room.get((int(x), int(y)))

    def get_room_color_by_room(self, room: tuple[int,int] | None):
        if room is None: return None
        meta = self.room_meta.get(room)
        return meta['color'] if meta else None

    def get_room_color_by_xy(self, x: int, y: int):
        return self.get_room_color_by_room(self.get_room_id_at(x, y))

    def get_visited_rooms_order(self) -> list[tuple[int,int]]:
        """Return discovery order as list of (col,row)."""
        return [rec['room'] for rec in self.visited_rooms_order]

    def get_rooms_layout_summary(self) -> dict:
        """Lightweight summary for debugging/visuals."""
        return {
            'rooms_in_row': self.rooms_in_row,
            'rooms_in_col': self.rooms_in_col,
            'n_rooms': len(self.room_meta),
            'n_corridor_cells': len(self.corridor_xy),
            'n_doors': len(self.door_xy),
            'first_visits': self.visited_rooms_order[:5]
        }
    # ---- END: room/corridor instrumentation helpers ---------------------------
    
    def Spawn_Obstacles(self, obstacle_rate= None):
        """
        Spawn grey 1×1 walls on coloured-room floor tiles while
        • leaving every corridor endpoint + its moat free,
        • ensuring the current agent corridor can still reach another corridor tile,
        • **never touching the agent's current position.**
        """
        print("DIS WHE")
        
        if obstacle_rate is None:
            obstacle_rate = self.obstacle_rate
        
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        # ───────── collect room & corridor tiles ───────────────────────────
        coloured_floor, corridor_cells = [], []
        for x in range(1, self.grid.width - 1):
            for y in range(1, self.grid.height - 1):
                cell = self.grid.get(x, y)
                if isinstance(cell, Floor):
                    (corridor_cells if cell.color == "black" else coloured_floor).append((x, y))

        assert len(corridor_cells) >= 2, "environment must contain corridors"

        # ───────── build protected set: corridor endpoints + moat ───────────
        def degree(coord):
            x, y = coord
            return sum(((x + dx, y + dy) in corridor_cells) for dx, dy in dirs)

        protected = {c for c in corridor_cells if degree(c) == 1}          # endpoints
        for x, y in list(protected):                                        # 1-tile moat
            protected.update({(x + dx, y + dy) for dx, dy in dirs})

        # Protect the agent's CURRENT position (not spawn position)
        protected.add(tuple(self.agent_pos))

        # remove protected tiles from candidate list
        coloured_floor = [c for c in coloured_floor if c not in protected]

        # ───────── identify current agent corridor tile (for connectivity test) ─────
        ax, ay = self.agent_pos
        current_corridor = min(corridor_cells, key=lambda c: abs(c[0] - ax) + abs(c[1] - ay))
        
        # ───── helper: can current corridor still reach another corridor? ────
        def corridor_reachable(blocked: set) -> bool:
            q = deque([current_corridor])
            visited = {current_corridor}

            while q:
                x, y = q.popleft()
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in visited or (nx, ny) in blocked:
                        continue
                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        cell = self.grid.get(nx, ny)
                        passable = (
                            cell is None
                            or (hasattr(cell, "can_overlap") and cell.can_overlap())
                            or isinstance(cell, Door)
                        )
                        if passable:
                            if (nx, ny) in corridor_cells and (nx, ny) != current_corridor:
                                return True       # found another corridor tile
                            visited.add((nx, ny))
                            q.append((nx, ny))
            return False                          # current corridor isolated

        # ───────────────────── greedy obstacle insertion ───────────────────
        random.shuffle(coloured_floor)
        target_n = int(len(coloured_floor) * obstacle_rate)
        new_obstacles = set()

        for (x, y) in coloured_floor:
            if len(new_obstacles) >= target_n:
                break
            candidate = self.current_obstacles | new_obstacles | {(x, y)}
            if corridor_reachable(candidate):
                new_obstacles.add((x, y))

        # ─────────────────────────── render walls ──────────────────────────
        for x, y in new_obstacles:
            self.grid.vert_wall(x, y, 1)
        
        # Update tracking
        self.current_obstacles.update(new_obstacles)

        if self.debug:
            print(f"Placed {len(new_obstacles)} new grey obstacles "
                f"(target {target_n}) – current corridor reachable & endpoints clear")
            print(f"Total obstacles now: {len(self.current_obstacles)}")
    def Remove_Obstacles(self):
        """Remove all current obstacles from the grid"""
        for x, y in self.current_obstacles:
            # Get the original floor color for this position
            # Find which room this position belongs to
            rooms_in_row = self.rooms_in_row
            rooms_in_col = self.rooms_in_col
            room_w = self.rooms_size
            room_h = self.rooms_size
            
            # Find which room this obstacle is in
            for row_inc in range(rooms_in_row):
                for col_inc in range(rooms_in_col):
                    xL = col_inc * (room_w + self.corridor_length)
                    yT = row_inc * (room_h + self.corridor_length)
                    xR = xL + room_w 
                    yB = yT + room_h
                    
                    if xL < x < xR and yT < y < yB:
                        # This obstacle is in this room, restore original color
                        color = list(self.color_idx.keys())[self._rand_int(0, 4)]
                        self.put_obj(Floor(color), x, y)
                        break
        
        self.current_obstacles.clear()
        if self.debug:
            print("Removed all obstacles")     
    def action_to_pose(self,action,current_pose):
        
        if action[0] == 1:
            current_pose[:2] = DIR_TO_VEC[current_pose[2]] + current_pose[:2]
        elif action[1] == 1:
            current_pose[2] = (current_pose[2]+1)%4
        elif action[2] == 1:
            current_pose[2] = (current_pose[2]-1)%4

        return current_pose


class AisleDoorFourTilesRoomsE(AisleDoorRooms):
    def __init__(self, size=120, rooms_size=5, rooms_in_row = 3, rooms_in_col = 3,  wT_around = 1, wT_size=1, agent_start_pos=None, corridor_length=4, max_steps = 4000):
        super().__init__(size=size, rooms_size=rooms_size, rooms_in_row = rooms_in_row, rooms_in_col = rooms_in_col,  wT_around = wT_around, wT_size=wT_size, agent_start_pos=agent_start_pos, corridor_length=corridor_length, max_steps = max_steps) #random_corridor_length if corridor_length 0 or not int
class AisleDoorFiveTilesRoomsE(AisleDoorRooms):
    def __init__(self,size=120, rooms_size=6, rooms_in_row = 3, rooms_in_col = 3,  wT_around = 1, wT_size=1, agent_start_pos=None, 
                 corridor_length=4, max_steps = 400):
        super().__init__(size=size, rooms_size=rooms_size, rooms_in_row = rooms_in_row, rooms_in_col = rooms_in_col,  \
                         wT_around = wT_around, wT_size=wT_size, agent_start_pos=agent_start_pos, corridor_length=corridor_length,\
                           max_steps = max_steps) 
class AisleDoorSixTilesRoomsE(AisleDoorRooms):
    def __init__(self,size=120, rooms_size=7, rooms_in_row = 3, rooms_in_col = 3,  wT_around = 1, wT_size=1, agent_start_pos=None,
                  corridor_length=4, max_steps = 400):
        super().__init__(size=size, rooms_size=rooms_size, rooms_in_row = rooms_in_row, rooms_in_col = rooms_in_col, wT_around = wT_around, wT_size=wT_size, agent_start_pos=agent_start_pos, corridor_length=corridor_length, max_steps = max_steps) 
class AisleDoorSevenTilesRoomsE(AisleDoorRooms):
    def __init__(self,size=120, rooms_size=8, rooms_in_row = 3, rooms_in_col = 3,  wT_around = 1, wT_size=1, agent_start_pos=None, 
                 corridor_length=4, max_steps = 400):
        super().__init__(size=size, rooms_size=rooms_size, rooms_in_row = rooms_in_row, rooms_in_col = rooms_in_col,  \
                         wT_around = wT_around, wT_size=wT_size, agent_start_pos=agent_start_pos, corridor_length=corridor_length, \
                         max_steps = max_steps) 
class AisleDoorEightTilesRoomsE(AisleDoorRooms):
    def __init__(self,size=120, rooms_size=9, rooms_in_row = 3, rooms_in_col = 3,  wT_around = 1, wT_size=1, agent_start_pos=None, 
                 corridor_length=4, max_steps = 40):
        super().__init__(size=size, rooms_size=rooms_size, rooms_in_row = rooms_in_row, rooms_in_col = rooms_in_col,  \
                         wT_around = wT_around, wT_size=wT_size, agent_start_pos=agent_start_pos, corridor_length=corridor_length, \
                         max_steps = max_steps) 
register(
    id='MiniGrid-4-tiles-ad-rooms-v0',
    entry_point='gym_minigrid.envs:AisleDoorFourTilesRoomsE'
)
register(
    id='MiniGrid-5-tiles-ad-rooms-v0',
    entry_point='gym_minigrid.envs:AisleDoorFiveTilesRoomsE'
)
register(
    id='MiniGrid-6-tiles-ad-rooms-v0',
    entry_point='gym_minigrid.envs:AisleDoorSixTilesRoomsE'
)
register(
    id='MiniGrid-7-tiles-ad-rooms-v0',
    entry_point='gym_minigrid.envs:AisleDoorSevenTilesRoomsE'
)
register(
    id='MiniGrid-8-tiles-ad-rooms-v0',
    entry_point='gym_minigrid.envs:AisleDoorEightTilesRoomsE'
)

class AisleDoorRoomsObstacles(MiniGridEnv):
    """
    Colored rooms connected by corridors *plus* random 1×1 grey obstacles.
    Rewards:
        +0.02  each safe step
        -1.0   on collision   → episode terminates
    Episode ends after 100 env steps if no collision occurs.
    """

    # ───────────────────────────────── init ──────────────────────────────────
    def __init__(
        self,
        size=120,
        agent_start_pos=None,
        agent_start_dir=0,
        rooms_size=5,
        rooms_in_row=3,
        rooms_in_col=3,
        corridor_length=5,
        automatic_door=True,
        wT_around=6,
        wT_size=1,
        obstacle_rate=0.15,          # ← 5 % of free cells become walls
        max_steps=100,               # ← horizon for “survival”
        debug=True,                 # ← print everything?
    ):
        # user-visible params
        self.automatic_door = automatic_door
        self.rooms_size = rooms_size
        self.rooms_in_row = rooms_in_row
        self.rooms_in_col = rooms_in_col
        self.obstacle_rate = obstacle_rate
        self.debug = debug
        self.wT_around=wT_around
        self.wT_size=wT_size
        self.size=size

        # store start pose if provided
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # allow fixed or random corridor length
        if isinstance(corridor_length, int) and corridor_length > 0:
            self.corridor_length = corridor_length
        else:
            self.corridor_length = self._rand_int(3, 7)

        # compute full grid size
        width = (rooms_in_col * rooms_size
                 + (rooms_in_col - 1) * (self.corridor_length + 1) - 1)
        height = (rooms_in_row * rooms_size
                  + (rooms_in_row - 1) * (self.corridor_length + 1) - 1)

        # small aesthetic tweaks so the grid isn’t exactly 20/30/40
        if width <= 20:
            width += 1
        if height <= 20:
            height += 1
        if width >= 30:
            width -= 1
        if height >= 30:
            height -= 1
        if width >= 40:
            width -= 1
        if height >= 40:
            height -= 1
        self.color_idx = {
            'red' : 0,
            'green' : 1,
            'blue'  : 2,
            'purple': 3,
            'white' : 4,
            
        }

        super().__init__(
            grid_size=None,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
        )

    # ─────────────────────────────── grid layout ─────────────────────────────
    def _gen_grid(self, width, height):
        if self.debug:
            print("Generating map …")

        super_grid_init = super()._gen_grid  # keep flake-8 calm
        print("4TFFFFFFFFF")       
        #define the number of room in col/row
        rooms_in_row = self.rooms_in_row
        rooms_in_col = self.rooms_in_col

        room_w = self.rooms_size
        room_h = self.rooms_size
       
        # Create an empty grid
        self.grid = Grid(width, height)
        self.grid.grid = [Floor('black')] * width * height

        # Generate the surrounding walls
    
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        balls_rooms = []
        while len(balls_rooms) < self.wT_around:
            col_room = self._rand_int(0, rooms_in_col)
            row_room = self._rand_int(0, rooms_in_row)
            if [col_room, row_room] not in balls_rooms:
                balls_rooms.append([col_room, row_room])

        pos_vert_R = None
        pos_horz_B = [None] * rooms_in_col
        agent_pose_options = []

     
        # For each row of rooms
        for row_inc in range(0, rooms_in_row):
            # For each column
            for col_inc in range(0, rooms_in_col):
                
                xL = col_inc * (room_w + self.corridor_length)
                yT = row_inc * (room_h + self.corridor_length)
                xR = xL + room_w 
                yB = yT + room_h
       
                color = list(self.color_idx.keys())[list(self.color_idx.values()).index(self._rand_int(0, 4))]

                for b in range(xL+1, xR):
                    for c in range(yT+1,yB):
                        self.put_obj(Floor(color),b, c)

                #upper wall and door
                if col_inc > 0:
                    self.grid.vert_wall(xL, yT, room_h+1)
                    self.grid.set(xL,pos_vert_R[1], Floor('black'))
                    self.put_obj(Door(color='white'), xL-int(self.corridor_length/2), pos_vert_R[1] )
                    corridor_depth = self._rand_int(0, 2)
                    agent_pose_options.append((xL-int(self.corridor_length/2)-corridor_depth,pos_vert_R[1],0))
    
                # Bottom wall and door
                if col_inc + 1 < rooms_in_col:
                    self.grid.vert_wall(xR, yT, room_h+1)
                    pos_vert_R = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos_vert_R, Floor('black'))
                    self.grid.horz_wall(xR, pos_vert_R[1]-1, self.corridor_length)  
                    self.grid.horz_wall(xR, pos_vert_R[1]+1, self.corridor_length)
                    corridor_depth = self._rand_int(0, 2)
                    agent_pose_options.append((xR+int(self.corridor_length/2)+corridor_depth, pos_vert_R[1],2))  

                if row_inc > 0:
                    self.grid.horz_wall(xL, yT, room_w+1)
                    self.grid.set(pos_horz_B[col_inc][0],yT, Floor('black'))
                    self.put_obj(Door(color='white'), pos_horz_B[col_inc][0] , yT-int(self.corridor_length/2))
                    corridor_depth = self._rand_int(0, 2)
                    agent_pose_options.append((pos_horz_B[col_inc][0],yT-int(self.corridor_length/2)-corridor_depth,1))
                   
                # Bottom wall and door
                if row_inc + 1 < rooms_in_row:
                    self.grid.horz_wall(xL, yB, room_w+1)
                    pos_horz_B[col_inc] = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos_horz_B[col_inc], Floor('black'))
                    self.grid.vert_wall(pos_horz_B[col_inc][0]-1, yB, self.corridor_length)  
                    self.grid.vert_wall(pos_horz_B[col_inc][0]+1, yB, self.corridor_length)  
                    corridor_depth = self._rand_int(0, 2)
                    
                    agent_pose_options.append((pos_horz_B[col_inc][0], yB+int(self.corridor_length/2)+corridor_depth, 3))  

         
        self.agent_pos, self.agent_dir = self._pick_random_spawn()
        self.vel_ob = [0,0]
        self.encoded_action = None      
        self._scatter_obstacles()
        spawn_cell = self.grid.get(*self.agent_pos)
        if isinstance(spawn_cell, Door):
            spawn_cell.toggle(self, self.agent_pos)
        elif isinstance(self.grid.get(*self.front_pos), Door):
            self.grid.get(*self.front_pos).toggle(self, self.front_pos)
        self.mission = "Avoid grey obstacles"

    # ------------------------------------------------------------------ #
    def _pick_random_spawn(self):
        """
        Return (np.array([x, y]), dir_idx) such that the start tile is

        • a coloured room floor   OR
        • a black floor that neighbours a coloured floor/door (corridor),
        • not on an obstacle / wall / goal,
        • and has at least one 4-neighbour it can step onto.
        """
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        candidates = []

        for x in range(1, self.grid.width - 1):
            for y in range(1, self.grid.height - 1):
                cell = self.grid.get(x, y)

                # we only ever spawn on plain Floor tiles
                if not isinstance(cell, Floor):
                    continue

                # ---------- colour logic ------------------------------------
                if cell.color == "black":
                    # Accept *only* if this black tile touches at least one
                    # coloured floor -- that marks it as a corridor segment
                    touches_room = False
                    for dx, dy in dirs:
                        ncell = self.grid.get(x + dx, y + dy)
                        if isinstance(ncell, Floor) and ncell.color != "black":
                            touches_room = True
                            break
                        if isinstance(ncell, Door):                # door counts
                            touches_room = True
                            break
                    if not touches_room:
                        continue        # outer-ring or isolated black tile → skip

                # ---------- boxed-in test -----------------------------------
                has_exit = False
                for dx, dy in dirs:
                    nx, ny  = x + dx, y + dy
                    ncell   = self.grid.get(nx, ny)
                    if (
                        ncell is None
                        or (hasattr(ncell, "can_overlap") and ncell.can_overlap())
                        or isinstance(ncell, Door)
                    ):
                        has_exit = True
                        break
                if not has_exit:
                    continue

                # passed all criteria
                candidates.append((x, y))

        assert candidates, "No valid spawn tiles after filtering!"

        x, y     = random.choice(candidates)
        dir_idx  = self._rand_int(0, 4)     # 0:→ 1:↓ 2:← 3:↑
        return np.array([x, y]), dir_idx
    # --------------------------------------------------------------------- #
    def _scatter_obstacles(self):
        """
        Drop grey 1×1 walls on coloured-room floor tiles while

        • leaving every corridor endpoint + its moat free,
        • ensuring the *spawn corridor* can still reach another corridor tile, and
        • **never touching the agent’s spawn tile itself.**
        """
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        # ───────── collect room & corridor tiles ───────────────────────────
        coloured_floor, corridor_cells = [], []
        for x in range(1, self.grid.width - 1):
            for y in range(1, self.grid.height - 1):
                cell = self.grid.get(x, y)
                if isinstance(cell, Floor):
                    (corridor_cells if cell.color == "black" else coloured_floor).append((x, y))

        assert len(corridor_cells) >= 2, "environment must contain corridors"

        # ───────── build protected set: corridor endpoints + moat ───────────
        def degree(coord):
            x, y = coord
            return sum(((x + dx, y + dy) in corridor_cells) for dx, dy in dirs)

        protected = {c for c in corridor_cells if degree(c) == 1}          # endpoints
        for x, y in list(protected):                                        # 1-tile moat
            protected.update({(x + dx, y + dy) for dx, dy in dirs})

        # NEW ─────—— also protect the agent’s spawn position ───────────────
        protected.add(tuple(self.agent_pos))

        # remove protected tiles from candidate list
        coloured_floor = [c for c in coloured_floor if c not in protected]

        # ───────── identify spawn-corridor tile (for connectivity test) ─────
        ax, ay = self.agent_pos
        spawn_corridor = min(corridor_cells, key=lambda c: abs(c[0] - ax) + abs(c[1] - ay))
            # ───── helper: can spawn corridor still reach another corridor? ────
        def spawn_corridor_ok(blocked: set) -> bool:
            q       = deque([spawn_corridor])
            visited = {spawn_corridor}

            while q:
                x, y = q.popleft()
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in visited or (nx, ny) in blocked:
                        continue
                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        cell = self.grid.get(nx, ny)
                        passable = (
                            cell is None
                            or (hasattr(cell, "can_overlap") and cell.can_overlap())
                            or isinstance(cell, Door)
                        )
                        if passable:
                            if (nx, ny) in corridor_cells and (nx, ny) != spawn_corridor:
                                return True       # found another corridor tile
                            visited.add((nx, ny))
                            q.append((nx, ny))
            return False                          # spawn corridor isolated

        # ───────────────────── greedy obstacle insertion ───────────────────
        random.shuffle(coloured_floor)
        target_n  = int(len(coloured_floor) * self.obstacle_rate)
        obstacles = set()

        for (x, y) in coloured_floor:
            if len(obstacles) >= target_n:
                break
            candidate = obstacles | {(x, y)}
            if spawn_corridor_ok(candidate):
                obstacles.add((x, y))

        # ─────────────────────────── render walls ──────────────────────────
        for x, y in obstacles:
            self.grid.vert_wall(x, y, 1)

        if self.debug:
            print(f"Placed {len(obstacles)} grey obstacles "
                f"(target {target_n}) – spawn corridor reachable & endpoints clear")
    # ─────────────────────────────── stepping ────────────────────────────────
    def step(self, action):
        """
        • Executes the primitive action via MiniGridEnv.step  
        • Opens automatic doors (old behaviour)  
        • Detects collisions and shapes reward  
        • Emits a boolean `info["collision"]`  
        • Terminates on collision or when max_steps is reached
        """
        # ------------------------------------------------------------------ #
        # 1) run the *real* env step (base reward ignored here)
        obs, _, done, info = super().step(action)
        # ------------------------------------------------------------------ #

        # ─────────────── automatic door helper ────────────────────────────
        if not done and self.automatic_door:
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell and fwd_cell.type == "door":
                # open the door, keep motion count unchanged
                obs2, _, _, _ = super().step(
                    self.actions.toggle, automatic_door=True
                )
                obs["door_open"] = obs2["door_open"]
                obs["image"] = obs2["image"]

        # ─────────────── collision detection / reward shaping ─────────────
        collided = (
            action == self.actions.forward and obs["vel_ob"] == [0, 0]
        )

        if collided:               # hit a wall or obstacle
            reward = -1.0
            done = True
        else:                      # survived one more step
            reward = 0.02

        info["collision"] = collided

        # ─────────────── bookkeeping for pose output (unchanged) ──────────
        real_action = [0, 0, 0] if obs["vel_ob"][0] == obs["vel_ob"][1] \
                               else obs["action"].copy()
        self.current_pose = self.action_to_pose(real_action, self.current_pose)
        obs["pose"] = self.current_pose

        if self.debug and (collided or done):
            flag = "COLLISION" if collided else "TIMEOUT"
            print(f"[step {self.step_count}] {flag} | reward {reward:.2f}")

        return obs, reward, done, info

    # ─────────────────────────── util for pose update ───────────────────────
    def action_to_pose(self, action, current_pose):
        if action[0] == 1:           # forward
            current_pose[:2] = DIR_TO_VEC[current_pose[2]] + current_pose[:2]
        elif action[1] == 1:         # turn right
            current_pose[2] = (current_pose[2] + 1) % 4
        elif action[2] == 1:         # turn left
            current_pose[2] = (current_pose[2] - 1) % 4
        return current_pose


# ─────────────────────────────── registry ───────────────────────────────────
register(
    id="MiniGrid-ADRooms-Collision-v0",
    entry_point="gym_minigrid.envs:AisleDoorRoomsObstacles",
)
