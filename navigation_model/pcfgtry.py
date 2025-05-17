from nltk import PCFG
from nltk.parse.generate import generate
import random
'''rule_grammar = PCFG.fromstring(""" 

    EXPLORE -> NAVPLAN [1.0000]
    NAVPLAN -> GOTO_9 [0.0000]
    NAVPLAN -> GOTO_8 [0.0000]
    NAVPLAN -> GOTO_10 [0.0000]
    NAVPLAN -> GOTO_14 [0.0000]
    NAVPLAN -> GOTO_11 [0.0000]
    NAVPLAN -> GOTO_7 [0.0000]
    NAVPLAN -> GOTO_13 [0.0000]
    NAVPLAN -> GOTO_15 [0.0000]
    NAVPLAN -> GOTO_2 [0.0000]
    NAVPLAN -> GOTO_5 [0.0000]
    NAVPLAN -> GOTO_0 [1.0000]
    NAVPLAN -> GOTO_6 [0.0000]
    NAVPLAN -> GOTO_3 [0.0000]
    NAVPLAN -> GOTO_12 [0.0000]
    NAVPLAN -> GOTO_4 [0.0000]
    NAVPLAN -> GOTO_1 [0.0000]
    GOTO_9 -> MOVESEQ_9_9 [1.0000]
    GOTO_8 -> MOVESEQ_9_8 [1.0000]
    GOTO_10 -> MOVESEQ_9_10 [1.0000]
    GOTO_14 -> MOVESEQ_9_14 [1.0000]
    GOTO_11 -> MOVESEQ_9_11 [1.0000]
    GOTO_7 -> MOVESEQ_9_7 [1.0000]
    GOTO_13 -> MOVESEQ_9_13 [1.0000]
    GOTO_15 -> MOVESEQ_9_15 [1.0000]
    GOTO_2 -> MOVESEQ_9_2 [1.0000]
    GOTO_5 -> MOVESEQ_9_5 [1.0000]
    GOTO_0 -> MOVESEQ_9_0 [1.0000]
    GOTO_6 -> MOVESEQ_9_6 [1.0000]
    GOTO_3 -> MOVESEQ_9_3 [1.0000]
    GOTO_12 -> MOVESEQ_9_12 [1.0000]
    GOTO_4 -> MOVESEQ_9_4 [1.0000]
    GOTO_1 -> MOVESEQ_9_1 [1.0000]
    MOVESEQ_9_9 -> STEP_9_9 [1.0000]
    MOVESEQ_9_8 -> STEP_9_8 [1.0000]
    MOVESEQ_9_10 -> STEP_9_10 [1.0000]
    MOVESEQ_9_14 -> STEP_9_14 [1.0000]
    MOVESEQ_9_11 -> STEP_9_11 [1.0000]
    MOVESEQ_9_7 -> STEP_9_7 [1.0000]
    MOVESEQ_9_13 -> STEP_9_13 [1.0000]
    MOVESEQ_9_15 -> STEP_9_15 [1.0000]
    MOVESEQ_9_2 -> STEP_9_2 [1.0000]
    MOVESEQ_9_5 -> STEP_9_5 [1.0000]
    MOVESEQ_9_0 -> STEP_9_0 [1.0000]
    MOVESEQ_9_6 -> STEP_9_6 [1.0000]
    MOVESEQ_9_3 -> STEP_9_3 [1.0000]
    MOVESEQ_9_12 -> STEP_9_12 [1.0000]
    MOVESEQ_9_4 -> STEP_9_4 [1.0000]
    MOVESEQ_9_1 -> STEP_9_1 [1.0000]
    MOVESEQ_9_18 -> STEP_9_19 STEP_19_18 [1.0000]
    STEP_9_5 -> 'step(9,5)' [1.0000]
    STEP_19_18 -> 'step(19,18)' [1.0000]
    STEP_9_14 -> 'step(9,14)' [1.0000]
    STEP_4_3 -> 'step(4,3)' [1.0000]
    STEP_9_12 -> 'step(9,12)' [1.0000]
    STEP_10_11 -> 'step(10,11)' [1.0000]
    STEP_5_6 -> 'step(5,6)' [1.0000]
    STEP_13_11 -> 'step(13,11)' [1.0000]
    STEP_9_8 -> 'step(9,8)' [1.0000]
    STEP_9_7 -> 'step(9,7)' [1.0000]
    STEP_9_0 -> 'step(9,0)' [1.0000]
    STEP_6_5 -> 'step(6,5)' [1.0000]
    STEP_9_11 -> 'step(9,11)' [1.0000]
    STEP_9_13 -> 'step(9,13)' [1.0000]
    STEP_3_5 -> 'step(3,5)' [1.0000]
    STEP_9_2 -> 'step(9,2)' [1.0000]
    STEP_2_3 -> 'step(2,3)' [1.0000]
    STEP_7_8 -> 'step(7,8)' [1.0000]
    STEP_15_14 -> 'step(15,14)' [1.0000]
    STEP_1_0 -> 'step(1,0)' [1.0000]
    STEP_9_1 -> 'step(9,1)' [1.0000]
    STEP_6_7 -> 'step(6,7)' [1.0000]
    STEP_9_6 -> 'step(9,6)' [1.0000]
    STEP_0_2 -> 'step(0,2)' [1.0000]
    STEP_14_13 -> 'step(14,13)' [1.0000]
    STEP_14_15 -> 'step(14,15)' [1.0000]
    STEP_8_7 -> 'step(8,7)' [1.0000]
    STEP_7_6 -> 'step(7,6)' [1.0000]
    STEP_3_4 -> 'step(3,4)' [1.0000]
    STEP_12_11 -> 'step(12,11)' [1.0000]
    STEP_9_15 -> 'step(9,15)' [1.0000]
    STEP_10_9 -> 'step(10,9)' [1.0000]
    STEP_9_3 -> 'step(9,3)' [1.0000]
    STEP_9_4 -> 'step(9,4)' [1.0000]
    STEP_0_1 -> 'step(0,1)' [1.0000]
    STEP_11_13 -> 'step(11,13)' [1.0000]
    STEP_9_19 -> 'step(9,19)' [1.0000]
    STEP_8_9 -> 'step(8,9)' [1.0000]
    STEP_2_0 -> 'step(2,0)' [1.0000]
    STEP_3_2 -> 'step(3,2)' [1.0000]
    STEP_9_9 -> 'step(9,9)' [1.0000]
    STEP_13_14 -> 'step(13,14)' [1.0000]
    STEP_9_10 -> 'step(9,10)' [1.0000]
    STEP_11_12 -> 'step(11,12)' [1.0000]
    STEP_11_10 -> 'step(11,10)' [1.0000]
    STEP_5_3 -> 'step(5,3)' [1.0000]
""")
def sample_plans(grammar, n=50):
    return list(generate(grammar, n=n, depth=100))

plans = sample_plans(rule_grammar)
print("Sampled plans:")
if not plans:
    print("  (No plans generated)")
else:
    for i, plan in enumerate(plans):
        print(f"  Plan {i+1}: {' '.join(plan)}")

def mutate_plan(plan):
    if not plan:
        return plan
    mutated = plan[:]
    swap_map = {"step(0,1)": "step(1,2)", "step(1,2)": "step(0,1)"}
    idx = random.randint(0, len(plan) - 1)
    if mutated[idx] in swap_map:
        mutated[idx] = swap_map[mutated[idx]]
    return mutated

print("\nMutated versions:")
for i, plan in enumerate(plans):
    mplan = mutate_plan(plan)
    print(f"  Plan {i+1} mutated: {' '.join(mplan)}")


from collections import deque

class LocalPrimitivePlanner:
    def __init__(self, step_controller, replay_buffer, memory_graph, window_size=5):
        self.env           = step_controller.env
        self.replay_buffer = replay_buffer
        self.mem_graph     = memory_graph
        self.win          = window_size  # how far in grid to search
    
    def get_primitives(self, u, v):
        # 1) Try replay buffer
        pb = self._prims_from_buffer(u, v)
        if pb:
            return pb
        # 2) Try greedy compass rollout
        cur_pose  = self.env.agent_pos + (self.env.agent_dir,)
        goal_pose = self.mem_graph.get_pose(v)
        cg = self.compass_prims(cur_pose, goal_pose, self.env)
        if self._reaches(cg, goal_pose):
            # cache and return
            self.mem_graph.add_prims(u, v, cg)
            return cg
        # 3) Fallback to BFS/A* in (x,y,θ)-space
        bf = self._bfs_prims(u, v, max_depth=20)
        self.mem_graph.add_prims(u, v, bf)
        return bf

    def _prims_from_buffer(self, u, v):
        # look for the most recent contiguous (u→v)
        for i in range(len(self.replay_buffer)-1, 0, -1):
            if (self.replay_buffer[i]['node_id']==u and
                self.replay_buffer[i+1]['node_id']==v):
                # collect the primitive actions between those entries
                return [self.replay_buffer[i]['action']]
        return []

    def _reaches(self, prims, goal_pose):
        # simulate forward in grid w/o affecting real env
        x,y,θ = self.env.agent_pos + (self.env.agent_dir,)
        dir_vecs = [(1,0),(0,1),(-1,0),(0,-1)]
        for a in prims:
            if a=='forward':
                dx,dy = dir_vecs[θ]
                if self.env.grid.get(int(x+dx), int(y+dy)).type=='wall':
                    return False
                x,y = x+dx, y+dy
            elif a=='left':
                θ = (θ-1)%4
            else:
                θ = (θ+1)%4
        return int(round(x))==goal_pose[0] and int(round(y))==goal_pose[1]
    
    def compass_prims(cur_pose, goal_pose, env, max_steps=20):
        """
        At each step:
        • Compute Δ = goal_xy − current_xy
        • Determine best turn (left/right) or forward that reduces ‖Δ‖
        • If that action would collide, try the other two
        • Repeat until you reach goal or exhaust max_steps
        """
        prims = []
        xg, yg = goal_pose[:2]
        x, y, θ = cur_pose
        for _ in range(max_steps):
            if int(round(x)) == xg and int(round(y)) == yg:
                break
            # Map θ to unit‐vector
            dir_vecs = [(1,0),(0,1),(-1,0),(0,-1)]
            dx, dy = dir_vecs[θ]
            # preferred: forward if it goes closer
            best = None
            best_dist = float('inf')
            for act in ['forward','left','right']:
                if act == 'forward':
                    nx, ny, nθ = x+dx, y+dy, θ
                elif act == 'left':
                    nx, ny, nθ = x, y, (θ-1)%4
                else:  # right
                    nx, ny, nθ = x, y, (θ+1)%4
                # check collision for forward
                if act=='forward' and env.grid.get(int(nx),int(ny)).type=='wall':
                    continue
                dist = (nx - xg)**2 + (ny - yg)**2
                if dist < best_dist:
                    best_dist, best, (x_,y_,θ_) = dist, act, (nx,ny,nθ)
            if best is None:
                break   # trapped
            prims.append(best)
            x, y, θ = x_, y_, θ_
        return prims


    def _bfs_prims(self, u, v, max_depth):
        # state = (x,y,θ), action_seq
        start = (*self.env.agent_pos, self.env.agent_dir)
        goal_xy = tuple(self.mem_graph.get_pose(v)[:2])
        dirs  = [(1,0),(0,1),(-1,0),(0,-1)]
        acts  = ['forward','left','right']
        dq    = deque([(start, [])])
        seen  = {start}
        while dq:
            (x,y,θ), seq = dq.popleft()
            if len(seq)>max_depth: continue
            if (int(round(x)),int(round(y)))==goal_xy:
                return seq
            for a in acts:
                if a=='forward':
                    dx,dy = dirs[θ]
                    nx,ny,nθ = x+dx, y+dy, θ
                    if self.env.grid.get(int(nx),int(ny)).type=='wall':
                        continue
                elif a=='left':
                    nx,ny,nθ = x,y,(θ-1)%4
                else:  # right
                    nx,ny,nθ = x,y,(θ+1)%4
                st = (nx,ny,nθ)
                if st in seen:
                    continue
                seen.add(st)
                dq.append((st, seq+[a]))
        # as a last resort, return empty sequence
        return []'''
from collections import deque, namedtuple
import heapq
import math

# 12×5 grid: 0=free, 1=wall
grid = [
    [0,0,0,1,0],
    [1,1,0,1,0],
    [0,0,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,1,1],                                                                                                                
    [0,1,1,1,1],
    [0,0,0,1,0],
    [1,1,0,1,0],
    [0,0,0,0,0],
]

# Heading to (dx,dy)
DIR_VECS = [(1,0),(0,1),(-1,0),(0,-1)]

State = namedtuple('State', ['x','y','d'])

def is_free(x,y):
    in_bounds = 0 <= x < 5 and 0 <= y < len(grid)
    free = in_bounds and grid[y][x]==0
    print(f"    [is_free] pos=({x},{y}) -> {'FREE' if free else 'WALL/OUT'}")
    return free

def heuristic(s, goal):
    # Manhattan + minimal turns to match goal heading
    h_pos = abs(s.x-goal.x)+abs(s.y-goal.y)
    δ = abs((s.d-goal.d)%4)
    h_rot = min(δ, 4-δ)
    h = h_pos + h_rot
    print(f"    [h] at {s} to {goal}: pos={h_pos}, rot={h_rot} -> h={h}")
    return h

def bfs_prims(start:State, goal:State) -> list[str]:
    """Breadth‐first search over (x,y,heading) space with debug."""
    print(f"[BFS] start={start} goal={goal}")
    visited = {start}
    queue = deque([(start, [])])
    step = 0
    while queue:
        (x,y,d), seq = queue.popleft()
        print(f"[BFS][{step}] pop {State(x,y,d)}  seq={seq}")
        if (x,y)==(goal.x,goal.y):
            print(f"[BFS] reached goal position at step {step} with seq={seq}")
            return seq
        for act in ['forward','left','right']:
            if act=='forward':
                dx,dy = DIR_VECS[d]
                nx,ny,nd = x+dx, y+dy, d
            elif act=='left':
                nx,ny,nd = x, y, (d-1)%4
            else:  # right
                nx,ny,nd = x, y, (d+1)%4
            ns = State(nx,ny,nd)
            print(f"    [BFS] try act={act} -> next={ns}", end='')
            if act=='forward' and not is_free(nx,ny):
                print("  SKIP (collision)")
                continue
            if ns in visited:
                print("  SKIP (visited)")
                continue
            visited.add(ns)
            new_seq = seq + [act]
            print(f"  ENQUEUE seq={new_seq}")
            queue.append((ns, new_seq))
        step += 1
    print("[BFS] no path found")
    return []

def astar_prims(start:State, goal:State) -> list[str]:
    """A* search with f = g + heuristic, debug‐instrumented."""
    print(f"[A*] start={start} goal={goal}")
    open_pq = [(heuristic(start,goal), 0, start, [])]
    g_score = {start: 0}
    closed = set()
    step = 0
    while open_pq:
        f, g, (x,y,d), seq = heapq.heappop(open_pq)
        print(f"[A*][{step}] POP  state={State(x,y,d)} g={g} f={f} seq={seq}")
        if (x,y,d) in closed:
            print("    [A*] skip (already closed)")
            step += 1
            continue
        closed.add((x,y,d))

        if (x,y)==(goal.x,goal.y):
            print(f"[A*] reached goal position at step {step} with seq={seq}")
            return seq

        for act in ['left','right','forward']:
            if act=='forward':
                dx,dy = DIR_VECS[d]
                nx,ny,nd = x+dx, y+dy, d
            elif act=='left':
                nx,ny,nd = x, y, (d-1)%4
            else:  # right
                nx,ny,nd = x, y, (d+1)%4

            ns = State(nx,ny,nd)
            print(f"    [A*] try act={act} -> next={ns}", end='')
            if act=='forward' and not is_free(nx,ny):
                print("  SKIP (collision)")
                continue
            if (nx,ny,nd) in closed:
                print("  SKIP (closed)")
                continue

            g2 = g + 1
            old_g = g_score.get(ns, float('inf'))
            if g2 < old_g:
                h2 = heuristic(ns, goal)
                f2 = g2 + h2
                g_score[ns] = g2
                new_seq = seq + [act]
                print(f"  PUSH  g={g2} h={h2} f={f2} seq={new_seq}")
                heapq.heappush(open_pq, (f2, g2, ns, new_seq))
            else:
                print(f"  SKIP (worse g: {g2} >= {old_g})")
        step += 1

    print("[A*] no path found")
    return []
def biased_astar(start:State,
                 goal:State,
                 desired_bearing:tuple[float,float],
                 bias_weight:float=0.5):
    """
    A* that prefers moves toward desired_bearing=(bx,by).
    """
    def heuristic(s):
        # basic manhattan + turns
        h_pos = abs(s.x-goal.x)+abs(s.y-goal.y)
        δ = abs((s.d-goal.d)%4)
        return h_pos + min(δ,4-δ)

    # priority queue holds (f, g, state, path)
    open_pq = [(heuristic(start), 0, start, [])]
    g_score = {start: 0}
    closed = set()

    # normalize bearing
    bx, by = desired_bearing
    norm = math.hypot(bx,by) or 1.0
    bx, by = bx/norm, by/norm

    while open_pq:
        f, g, (x,y,d), path = heapq.heappop(open_pq)
        if (x,y,d) in closed:
            continue
        closed.add((x,y,d))
        if (x,y)==(goal.x,goal.y):
            return path

        for act in ['forward','left','right']:
            if act=='forward':
                dx,dy = DIR_VECS[d]
                nx,ny,nd = x+dx, y+dy, d
            elif act=='left':
                nx,ny,nd = x, y, (d-1)%4
                dx,dy = DIR_VECS[nd]
            else:  # right
                nx,ny,nd = x, y, (d+1)%4
                dx,dy = DIR_VECS[nd]

            # collision check omitted for brevity...
            ns = State(nx,ny,nd)

            # directional penalty: how well does (dx,dy) align with (bx,by)?
            cosθ = dx*bx + dy*by
            dir_pen = bias_weight * (1 - cosθ)

            step_cost = 1 + dir_pen
            g2 = g + step_cost

            if g2 < g_score.get(ns, float('inf')):
                g_score[ns] = g2
                heapq.heappush(open_pq, (g2 + heuristic(ns), g2, ns, path+[act]))

    return []  # no path found

# Example run
start = State(0,0,1)   # at (0,0), facing South
goal  = State(4,6,2)   # at (3,2), facing West

print("\n--- Running BFS ---")
bfs_path   = bfs_prims(start, goal)

print("\n--- Running A* ---")
astar_path = astar_prims(start, goal)

print("\nFinal Results:")
print(f"BFS sequence   ({len(bfs_path)} steps): {bfs_path}")
print(f"A*  sequence   ({len(astar_path)} steps): {astar_path}")
