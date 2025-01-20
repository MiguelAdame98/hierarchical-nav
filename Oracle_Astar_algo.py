import numpy as np
import gym
from gym_minigrid.wrappers import *

class Oracle_path(gym.core.ObservationWrapper):
    """
    Extract the Goal in Env and store them to count the shortest number of steps to reach objective
    """
    def __init__(self, env):
        super().__init__(env)
        self.goal_position = []
        self.door_poses_in_env = {}
        n_rooms = self.rooms_size + self.corridor_length
        room_threshold = self.rooms_size/n_rooms
        #print('rooms in row and col?', self.rooms_in_row, self.rooms_in_col)
        self.maze = [[0 for _ in range(self.height)] for _ in range(self.width)]
        if not self.goal_position:
            for pose,tile in enumerate(self.grid.grid):
                xi, yi= pose % self.width, pose // self.width
                self.maze[xi][yi] = tile
                if isinstance(tile,(Goal)):
                    self.goal_position.append((xi, yi))
                if tile.type == 'door':
                    room = [int(np.floor(xi/n_rooms)), int(np.floor(yi/n_rooms))]
                    room2 = room.copy()
                    if (xi/n_rooms) % 1 >= room_threshold and room[0] < self.rooms_in_col-1:
                        room2[0] +=1
                    if (yi/n_rooms) % 1 >= room_threshold and room[1] < self.rooms_in_row-1:
                        room2[1] +=1
                    
                    room = tuple(room)
                    room2 = tuple(room2)

                    #print('xi, yi', xi, yi, room, room2)
                    
                    if room not in self.door_poses_in_env:
                        self.door_poses_in_env[room] = []

                    if room2 not in self.door_poses_in_env:
                        self.door_poses_in_env[room2] = []

                    #print(self.door_poses_in_env)
                    self.door_poses_in_env[room].append((xi,yi))
                    self.door_poses_in_env[room2].append((xi,yi))

                    
        for room in self.door_poses_in_env.keys():
            doors = np.array(self.door_poses_in_env[room])
            #doors: R,D,L,UP
            self.door_poses_in_env[room] = [max(doors, key=lambda x: x[0]), max(doors, key=lambda x: x[1]),
                                            min(doors, key=lambda x: x[0]), min(doors, key=lambda x: x[1]), 
                                             ]

        # print('Oracle Goals ', self.goal_position)
        # print('in oracle init door_poses_in_env',self.door_poses_in_env)
        # self.steps_to_explore_env()
        # m = self.steps_to_closest_goal()
    def get_doors_poses_in_env(self):
        return self.door_poses_in_env
    
    def steps_to_explore_env(self):
        
        agent_pos = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_dir])

        full_path_length = 0
        full_path = []
        visited_rooms = 1
        self.door_poses_in_env_exploration = self.door_poses_in_env.copy()
        while len(self.door_poses_in_env_exploration.keys()) > 1 and visited_rooms < self.rooms_in_row* self.rooms_in_col +1 :
            #print(visited_rooms)
            path, path_length, agent_pos = self.go_to_next_room(agent_pos)
            full_path_length += path_length
            visited_rooms += 1
            full_path.extend(path)
        path_length+= self.corridor_length + 5 #goal is door step, so we enter room + turn around

        return full_path, full_path_length
    
    def go_to_next_room(self, agent_pos):
        n_rooms = self.rooms_size + self.corridor_length
        dir_room_vector = [
            # Pointing right 
            np.array([1,0]),
            # Pointing down  
            np.array([0, 1]),
            # POinting left
            np.array([-1, 0]),
            # Pointing up
            np.array([0, -1]),
            ] 
        start_room = [int(np.floor(agent_pos[0]/n_rooms)), int(np.floor(agent_pos[1]/n_rooms))]

        #if orientation 0 or 1 then we go +1 in rooms
        if agent_pos[2] < 2 :
            #print(start_room)
            start_room = start_room + dir_room_vector[agent_pos[2]]

        adjacent_rooms = []
        for vector in dir_room_vector:
            room = start_room + vector
            #print(room, 0 <= room[0] < self.rooms_in_row, 0<= room[1] < self.rooms_in_col)
            if  0 <= room[0] < self.rooms_in_col and 0<= room[1] < self.rooms_in_row:
                adjacent_rooms.append(room)
        
        #order the adjacent room from smallest room value with borders coming first
        adjacent_rooms.sort(key=lambda x: (x[0]+ x[1]))
        adjacent_rooms =  np.array(adjacent_rooms)
        min_val_idx = np.where(adjacent_rooms == np.min([adjacent_rooms[:,0], adjacent_rooms[:,1]]))[0]
        #print('adjacent_rooms', adjacent_rooms ,'and min value index in it', min_val_idx)
        
        if len(min_val_idx) ==1 and min_val_idx[0] == 1:
            tmp = adjacent_rooms[0].copy()
            adjacent_rooms[0] = adjacent_rooms[1]
            adjacent_rooms[1] = tmp
        else :
            for i in range(1,len(min_val_idx)):
                #if NOT following or same numbers
                if not abs(min_val_idx[i-1] - min_val_idx[i]) <= 1 :
                    idx = min_val_idx[i]
                    tmp = adjacent_rooms[idx-1].copy()
                    adjacent_rooms[idx-1] = adjacent_rooms[idx]
                    adjacent_rooms[idx] = tmp
        
        #print('current room', start_room, 'adjacent_rooms',adjacent_rooms)
        
        for room in adjacent_rooms:
            #print(np.array(room- start_room), dir_room_vector)
            dir_vector_idx =  np.where((dir_room_vector == np.array(room- start_room)).all(axis=1))[0][0]
            room = tuple(room)
            if room not in self.door_poses_in_env_exploration:
                #print('room', room, 'not reachable')
                continue
            start_room = tuple(start_room)
            door_goal = self.door_poses_in_env_exploration[start_room][dir_vector_idx]
            #print('GOAL room', room, 'goal door', door_goal, 'direction', dir_vector_idx)
            break

        path = astar(self.maze, tuple(agent_pos), tuple(door_goal))
        # Calculate number of turns in path
        path_length, num_turns = include_turns_in_path_length(path)
        #print('The oracle said')
        # print('path len',len(path)-1)
        # print('n_turns', num_turns)
        # for node in path:
        #     print(node.position, node.direction)
        
        if path_length == 0: #if we are at door 
            path_length+= self.rooms_size*2
        else:
            path_length+=4
        #print('num steps:', path_length)
        agent_pos = np.array([door_goal[0], door_goal[1],dir_vector_idx])
        
        count_adjacent_rooms = 0
        for vector in dir_room_vector:
            next_room = room + vector
            #print(next_room, 0 <= next_room[0] < self.rooms_in_row, 0<= next_room[1] < self.rooms_in_col)
            if tuple(next_room) in self.door_poses_in_env_exploration:
                count_adjacent_rooms+= 1 
        #print('count_adjacent_rooms of room', room, count_adjacent_rooms)
        if count_adjacent_rooms > 1:
            #print('deleting', start_room)
            del self.door_poses_in_env_exploration[start_room]
        # print('===========')
        # print()
        return path, path_length, agent_pos
    

    def steps_to_closest_goal(self):
        agent_pos = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_dir])
        
        #TODO: to review when multiple goal #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all() (shouldn't exist)
        self.goal_position.sort(key=lambda i: sort_by_distance(i, agent_pos[:2]))
        
        # print(self.goal_position)
        path = astar(self.maze, tuple(agent_pos), tuple(self.goal_position[0]))
        # Calculate number of turns in path
        path_length, num_turns = include_turns_in_path_length(path)
        print('The oracle said')
        print('path len',len(path)-1)
        print('n_turns', num_turns)
        # for node in path:
        #     print(node.position, node.direction)
        print('num steps:', path_length)
        
        return path, path_length -1

def include_turns_in_path_length(path):
    num_turns = 0
    for i in range(1, len(path)):
        if path[i].direction != path[i-1].direction:
            # print('turn',path[i-1].position,path[i].position)
            # print('dir', path[i-1].direction, path[i].direction)
            num_turns += 1
            #If we do a 180'
            if np.all(abs(np.array(path[i-1].direction)) - abs(np.array(path[i].direction)) == [0,0]):
                num_turns += 1
    
    return len(path)-1 + num_turns, num_turns
    
def sort_by_distance(goal, start):
    goal_dist = np.linalg.norm([goal  - start])
    return goal_dist

class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None, dir= None):

        if dir is not None:
            dir_vector = [
            # Pointing right (positive X)
            (1, 0),
            # Down (positive Y)
            (0, 1),
            # Pointing left (negative X)
            (-1, 0),
            # Up (negative Y)
            (0, -1),
            ] 
            dir = dir_vector[dir]
        self.parent = parent
        self.position = position
        self.direction = dir  # added to store direction from parent

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""       
    # Create start and end node
    
    start_node = Node(None, start[:2], dir=start[2])
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        
        # Get the current node
        # if end_node in open_list:
        #     current_index = open_list.index(end_node)
        #     current_node = open_list[current_index]
        # else:
        
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # del current off open list, add to closed list
        del open_list[current_index]
        closed_list.append(current_node)
        #print('current node',current_node.position)
        # Found the goal
        if current_node == end_node:
            #print('found goal')
            path = []
            current = current_node

            while current is not None:
                path.append(current)
                current = current.parent

            
            return path[::-1]  # Return reversed path 

        #This part actually ensure we first try to go straight before turning
        directions_next_node = [(0, -1), (-1, 0), (0,1) ,(1, 0)]
        # if current_node.direction != None:
        #     directions_next_node.remove(current_node.direction)
        #     directions_next_node.insert(0,current_node.direction )
            #print(current_node.position, current_node.direction, directions_next_node)
        for new_position in directions_next_node:  # Only allow orthogonal movement

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range (if no walls to delimit plan)
            # if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[0]) -1) or node_position[1] < 0:
            #     continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]].type == 'wall':
                #print('wall at position', node_position[0], node_position[1])
                continue

            # Create new node
            new_node = Node(current_node, node_position)
            
            # Store direction from parent
            #if current_node.parent:
            x_diff = new_node.position[0] - current_node.position[0]
            y_diff = new_node.position[1] - current_node.position[1]
            # print('xdiff and y diff of new node', current_node.position, new_node.position, x_diff, y_diff)
            if x_diff > 0:
                new_node.direction = (0,1)
            elif x_diff < 0:
                new_node.direction = (0,-1)
            elif y_diff > 0:
                new_node.direction = (1,0)
            elif y_diff < 0:
                new_node.direction = (-1,0)

            # Calculate f, g, and h values
            t = 0
            #if we need to turn, we add a weigth to it
            if current_node.direction != new_node.direction: 
                t+=1
            new_node.g = current_node.g + 1 + t
            new_node.h = abs(new_node.position[0] - end_node.position[0]) + abs(new_node.position[1] - end_node.position[1])
            #print('new_node.h', new_node.position, new_node.h )
            new_node.f = new_node.g + new_node.h

            # Check if node is already in closed list
            for closed_node in closed_list:
                if new_node == closed_node:
                    continue

            # Check if node is already in open list
            for open_node in open_list:
                if new_node == open_node and new_node.g > open_node.g:
                    continue

            # Add the new node to the open list
            #print('show me the appended node', new_node.position)
            open_list.append(new_node)

def main():
    #test maze must be minigrid env now
    # maze = [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0,1],
    #         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0,1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1],
    #         [1, 1, 0, 1, 1, 1, 1, 0, 1, 1,1],
    #         [0, 1, 0, 1, 1, 0, 1, 0, 1, 0,1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1],
    #         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0,1],
    #         [0, 0, 0, 0, 1, 0, 0., 0, 0, 0,1],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1]]

    start = (0, 0)
    end = (7, 6)

    path, n_turns = astar(maze, start, end)
    print('path len',len(path))
    for node in path:
        #print('node',node)
        print(node.position)


if __name__ == '__main__':
    main()