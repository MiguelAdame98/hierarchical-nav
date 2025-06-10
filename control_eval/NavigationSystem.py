
import time
import random
import numpy as np
from collections import deque
from collections import defaultdict
from nltk import PCFG
import math
import heapq
import torch
from collections import namedtuple
from nltk.parse.generate import generate
State = namedtuple("State", ["x","y","d"])
DIR_VECS = [(1,0),(0,1),(-1,0),(0,-1)]
ACTIONS   = ["forward","left","right"]

class NavigationSystem:
    def __init__(self, memory_graph, get_current_pose_func):
        self.memory_graph      = memory_graph          # live reference
        self.get_current_pose  = get_current_pose_func # lambda ↦ (x,y,d)

        # ─ plan bookkeeping ─
        self.full_plan_tokens: list[tuple[int, int]] = []   # [(u,v), …]
        self.token_idx   = 0        # which edge
        self.token_prims: list[str] = []
        self.prim_idx    = 0
        self.plan_progress = -1.0   # -1=no plan, 0-1 executing, 1.1 done

        # ─ misc state you already had ─
        self.target_node_id   = None
        self.navigation_flags = {}
        
    def get_available_nodes_with_confidence(self, min_confidence=0.5):
        """Get nodes that are viable for navigation"""
        viable_nodes = []
        emap = self.memory_graph.experience_map
        
        for exp in emap.exps:
            # Check if node has sufficient confidence
            if hasattr(exp, 'confidence') and exp.confidence >= min_confidence:
                viable_nodes.append({
                    'id': exp.id,
                    'confidence': exp.confidence,
                    'pose': (exp.x_m, exp.y_m, getattr(exp, 'dir', 0))
                })
        
        return viable_nodes
    
    def check_navigation_feasibility(self):
        """Early check for navigation feasibility"""
        viable_nodes = self.get_available_nodes_with_confidence()
        
        # Check edge cases
        if len(viable_nodes) <= 2:
            self.navigation_flags['insufficient_nodes'] = True
            return False
            
        if len(viable_nodes) == 0:
            self.navigation_flags['no_viable_nodes'] = True
            return False
            
        # Check if we have a current plan that's still valid
        if self.current_full_plan and self.target_node_id:
            target_still_viable = any(n['id'] == self.target_node_id for n in viable_nodes)
            if not target_still_viable:
                self.navigation_flags['target_became_unviable'] = True
                return False
                
        self.navigation_flags.clear()  # Clear flags if everything is okay
        return True
    
    def generate_plan_with_pcfg(self, grammar: PCFG):
        """
        Sample one sentence from `grammar`, tokenise it into a list
        of (u,v) edges, reset all plan indices, and store the final
        target node ID.  Returns the list of edge tuples.
        """
        sentence = random.choice(list(generate(grammar, depth=50)))  # list[str]
        plan_str = " ".join(sentence)

        # Tokenise into edge tuples
        self.full_plan_tokens = self._tokenize_plan(plan_str)
        self.token_idx = self.prim_idx = 0
        self.token_prims.clear()
        self.plan_progress = 0.0

        # Final target node = second element of last edge
        if self.full_plan_tokens:
            _, self.target_node_id = self.full_plan_tokens[-1]
        else:
            self.target_node_id = None            # edge case: empty plan

        return self.full_plan_tokens            # caller rarely needs the string
    
    def _tokenize_plan(self, full_plan: str) -> list[tuple[int, int]]:
        """Convert 'STEP_9_8 STEP_8_5 …' → [(9,8), (8,5), …]."""
        return [
            tuple(map(int, t.split('_')[1:3]))
            for t in full_plan.strip().split()
            if t.startswith('STEP_')
        ]
    def _load_edge_primitives(self):
        """Ensure self.token_prims is filled for the current edge."""
        if self.token_idx >= len(self.full_plan_tokens):
            self.token_prims = []
            return
        if self.prim_idx == 0 and not self.token_prims:          # first entry
            u, v = self.full_plan_tokens[self.token_idx]
            self.token_prims = self.get_primitives(u, v) or []
    
    def step_plan(self) -> tuple[list[str], int]:
        """
        Returns ([prim], 1) or ([],0) if finished / no plan.
        Automatically advances internal indices.
        """
        # finished or no plan
        if self.token_idx >= len(self.full_plan_tokens):
            self.plan_progress = 1.1 if self.full_plan_tokens else -1.0
            return [], 0

        # ensure primitives for current edge
        self._load_edge_primitives()
        if not self.token_prims:                 # edge unreachable
            self.plan_progress = -1.0
            return [], 0

        # serve one primitive
        prim = self.token_prims[self.prim_idx]
        self.prim_idx += 1

        # edge completed?
        if self.prim_idx >= len(self.token_prims):
            self.token_idx += 1
            self.token_prims = []
            self.prim_idx = 0

        # progress scalar 0…1
        done_edges = self.token_idx
        fraction   = self.prim_idx / max(len(self.token_prims), 1) if self.token_prims else 0
        self.plan_progress = (done_edges + fraction) / len(self.full_plan_tokens)
        return [prim], 1

    def new_plan_from_grammar(self, grammar: PCFG):
        sent = random.choice(list(generate(grammar, depth=50)))
        self.full_plan_tokens = self._tokenize_plan(" ".join(sent))
        self.token_idx = self.prim_idx = 0
        self.token_prims.clear()
        self.plan_progress = 0.0
        # final target = last rhs node
        if self.full_plan_tokens:
            _, self.target_node_id = self.full_plan_tokens[-1]

    def progress_scalar(self) -> float:
        """-1 no plan, 0-1 executing, 1.1 finished."""
        return self.plan_progress
    def get_next_navigation_step(self):
        """Get the next step in the current plan"""
        if (self.current_plan_tokens and 
            self.current_token_index < len(self.current_plan_tokens)):
            
            current_token = self.current_plan_tokens[self.current_token_index]
            self.current_token_index += 1
            
            # Parse token to get source and target nodes
            # "STEP_9_8" -> source=9, target=8
            parts = current_token.split('_')
            if len(parts) >= 3:
                source_node = int(parts[1])
                target_node = int(parts[2])
                return source_node, target_node
                
        return None, None
    
    def universal_navigation(self, submode: str) -> tuple[list[str], int]:
        cur_pose = self.get_current_pose()

        if submode == "plan_following":
            return self.step_plan()

        elif submode == "evade":
            # one-shot detour primitive towards current edge target
            if not self.full_plan_tokens:
                return [], 0
            _, tgt = self.full_plan_tokens[self.token_idx]
            tgt_pose = self.memory_graph.experience_map.get_pose(tgt)
            start = self._pose_to_state(cur_pose)
            goal  = self._pose_to_state(tgt_pose)
            detour = self.astar_prims(start, goal, None, 5)
            # return first primitive; remaining will be re-computed next tick
            return (detour[:1], 1) if detour else ([], 0)

        elif submode == "replan":
            # penalise current target node
            if self.token_idx < len(self.full_plan_tokens):
                _, bad = self.full_plan_tokens[self.token_idx]
                self._crash_node_confidence(bad)

            # rebuild PCFG with updated confidences
            grammar = self.build_pcfg_from_memory()           # already confidence-aware
            self.new_plan_from_grammar(grammar)
            return self.step_plan()

        else:
            raise ValueError(submode)
    
    def _handle_plan_following(self, current_pose, objective):
        """Handle normal plan following"""
        if isinstance(objective, int):  # objective is a node_id
            # Get primitives using your existing link-based system
            source_node, target_node = self.get_next_navigation_step()
            if source_node is not None and target_node is not None:
                primitives = self.get_primitives(source_node, target_node)
                return primitives, len(primitives)
        else:  # objective is a pose
            # Use A* for pose-to-pose navigation
            start_state = self._pose_to_state(current_pose)
            goal_state = self._pose_to_state(objective)
            primitives = self.astar_prims(start_state, goal_state, 
                                        egocentric_process=None, num_samples=5)
            return primitives, len(primitives)
            
        return [], 0
    
    def _handle_evade(self, current_pose, objective):
        """Handle obstacle evasion"""
        # Use A* to find path around obstacles
        start_state = self._pose_to_state(current_pose)
        
        if isinstance(objective, int):
            # Get pose of target node
            emap = self.memory_graph.experience_map
            target_pose = emap.get_pose(objective)
            goal_state = self._pose_to_state(target_pose)
        else:
            goal_state = self._pose_to_state(objective)
            
        # A* should naturally avoid obstacles
        primitives = self.astar_prims(start_state, goal_state,
                                    egocentric_process=None, num_samples=5)
        return primitives, len(primitives)
    
    def _handle_replan(self, current_pose, objective):
        """Handle replanning when current objective becomes unreachable"""
        if isinstance(objective, int):
            # Crash confidence of the problematic node
            self._crash_node_confidence(objective)
            
        # Generate new plan with updated confidence values
        grammar = self._rebuild_pcfg_with_updated_confidence()
        new_plan = self.generate_plan_with_pcfg(grammar)
        
        # Execute first step of new plan
        return self._handle_plan_following(current_pose, self.target_node_id)
    
    def _crash_node_confidence(self, node_id):
        """Reduce confidence of a problematic node"""
        emap = self.memory_graph.experience_map
        for exp in emap.exps:
            if exp.id == node_id:
                if hasattr(exp, 'confidence'):
                    exp.confidence *= 0.5  # Reduce confidence by half
                else:
                    exp.confidence = 0.1  # Set low confidence
                break

    def generate_initial_plan(self, grammar: PCFG):
        """1-shot helper for the controller (kept for API parity)."""
        self.new_plan_from_grammar(grammar)

    def get_observer_data(self):
        """Provide data for the navigation observer"""
        current_pose = self.get_current_pose()
        
        observer_data = {
            'full_plan': self.current_full_plan,
            'plan_tokens': self.current_plan_tokens,
            'current_token_index': self.current_token_index,
            'target_node_id': self.target_node_id,
            'current_pose': current_pose,
            'navigation_flags': self.navigation_flags.copy(),
            'plan_progress': self.current_token_index / max(len(self.current_plan_tokens), 1)
        }
        
        return observer_data
    
    def _pose_to_state(self, pose):
        """Convert pose tuple to State object"""
        # Assuming State class exists as in your A* implementation
        x, y, direction = pose
        return State(int(round(x)), int(round(y)), int(direction))
    
    def _sample_from_pcfg(self, grammar):
        """Sample a plan from the PCFG"""
        # Your existing PCFG sampling logic here
        # This is a placeholder - use your actual implementation
        pass
    
    def _extract_target_from_plan(self, plan):
        """Extract target node ID from sampled plan"""
        # Parse the plan string to find the ultimate target
        # This depends on your PCFG structure
        pass
    
    def _rebuild_pcfg_with_updated_confidence(self):
        """Rebuild PCFG with updated node confidence values"""
        # Your existing build_pcfg_from_memory but considering confidence
        pass

# Observer integration (to be added to your step function)
    def plan_progress_placeholder(self):
        """
        Enhanced plan progress observer for HMM
        """
        if hasattr(self, 'nav_system'):
            observer_data = self.nav_system.get_observer_data()
            
            # Calculate various metrics for HMM
            progress_metrics = {
                'plan_completion': observer_data['plan_progress'],
                'stuck_flag': 'insufficient_nodes' in observer_data['navigation_flags'],
                'replan_needed': 'target_became_unviable' in observer_data['navigation_flags'],
                'no_path_flag': 'no_viable_nodes' in observer_data['navigation_flags']
            }
            
            return progress_metrics
        
        return {'plan_completion': 0.0, 'stuck_flag': False, 'replan_needed': False, 'no_path_flag': False}

    def build_pcfg_from_memory(self):
        mg      = self.memory_graph
        emap    = mg.experience_map
        current = mg.get_current_exp_id()
        exps_by_dist = mg.get_exps_organised(current)
        all_exps     = emap.exps

        # 1) Abstract graph & distance priors
        graph = {e.id: [l.target.id for l in e.links] for e in all_exps}
        print("[PCFG DEBUG] graph:", graph)

        id_to_dist  = {}
        total_w = 0.0
        for d in exps_by_dist:
            dist = math.hypot(d['x'], d['y'])
            id_to_dist[d['id']] = dist
            total_w += 1.0/(dist + 1e-5)

        # 2) Top‐level: EXPLORE→NAVPLAN and NAVPLAN→GOTOₜ
        rules = defaultdict(list)
        rules['EXPLORE'].append(('NAVPLAN', 1.0))
        for tgt, dist in id_to_dist.items():
            p = (1.0/(dist+1e-5)) / total_w
            rules['NAVPLAN'].append((f'GOTO_{tgt}', p))
            rules[f'GOTO_{tgt}'].append((f'MOVESEQ_{current}_{tgt}', 1.0))

        # 3) BFS helper on abstract graph
        def find_paths(start, goal, max_depth=15, max_paths=10):
            paths, q = [], deque([[start]])
            while q and len(paths)<max_paths:
                path = q.popleft()
                if path[-1]==goal:
                    paths.append(path)
                elif len(path)<max_depth:
                    for nb in graph.get(path[-1],[]):
                        if nb not in path:
                            q.append(path+[nb])
            return paths

        # 4) Gather all abstract paths and their edges
        hop_edges = set()
        hopseqs   = {}
        for tgt in id_to_dist:
            paths = find_paths(current, tgt)
            hopseqs[tgt] = paths
            for path in paths:
                for u,v in zip(path, path[1:]):
                    hop_edges.add((u,v))
        hop_edges.add((current, current))

        # 4a) MOVESEQ_current→tgt → HOPSEQ_current→tgt
        for tgt in id_to_dist:
            lhs = f'MOVESEQ_{current}_{tgt}'
            rules[lhs].append((f'HOPSEQ_{current}_{tgt}', 1.0))

        # 4b) HOPSEQ_current→tgt → STEP_u_v … but *prefix* (current→current)
        for tgt, paths in hopseqs.items():
            lhs = f'HOPSEQ_{current}_{tgt}'
            if not paths and current!=tgt:
                # no path found
                rules[lhs].append((f'STEP_{current}_{tgt}', 1.0))
                hop_edges.add((current, tgt))
            else:
                w = 1.0/len(paths) if paths else 1.0
                for path in paths:
                    # *** here’s the dummy “first hop” ***
                    hops = [(current, current)] + list(zip(path, path[1:]))
                    seq = [f'STEP_{u}_{v}' for u,v in hops]
                    rhs = " ".join(seq)
                    rules[lhs].append((rhs, w))
                    print(f"[PCFG DEBUG] HOPSEQ {lhs} ← {hops} → {seq}")

        # 5) STEP_u_v → primitives OR fallback
        for (u,v) in hop_edges:
            lhs = f'STEP_{u}_{v}'
            prims = self.get_primitives(u, v)
            if prims:
                rhs = " ".join(f"'{p}'" for p in prims)
                rules[lhs].append((rhs, 1.0))
                print(f"[PCFG DEBUG] STEP_{u}_{v} → prims {prims}")
            else:
                rules[lhs].append((f"'step({u},{v})'", 1.0))
                print(f"[PCFG DEBUG] STEP_{u}_{v} → fallback 'step({u},{v})'")

        # 6) Optional hard‐coded extras
        hard = {
        f'MOVESEQ_{current}_18': [f"step({current},19)","step(19,18)"],
        f'MOVESEQ_{current}_3' : [f"step({current},5)","step(5,4)","step(4,3)"],
        }
        for lhs, steps in hard.items():
            if lhs not in rules:
                rhs = " ".join(f"'{s}'" for s in steps)
                rules[lhs].append((rhs, 1.0))
                print(f"[PCFG DEBUG] hardcoded {lhs} → {steps}")

        # 7) Assemble into PCFG
        pcfg_lines = []
        for lhs, prods in rules.items():
            total = sum(p for _,p in prods)
            for rhs,p in prods:
                pcfg_lines.append(f"{lhs} -> {rhs} [{p/total:.4f}]")

        grammar_src = "\n".join(pcfg_lines)
        print("[PCFG DEBUG] Final grammar:\n" + grammar_src)
        return PCFG.fromstring(grammar_src)
          
    def get_primitives(self, u: int, v: int) -> list[str]:
        """
        Return the best primitive sequence for traversing the edge u→v:

        0) If we’ve already stored a path in the ExperienceLink, return it.
        1) Otherwise, build a fresh A* plan from *our real pose* → node v.
        2) Cache it in the link and return.
        """
        emap = self.memory_graph.experience_map

        # 0) do we already have a stored primitive path?
        link = self._find_link(u, v)
        if link and link.path_forward:
            print(f"[get_primitives] cached path for {u}->{v}: {link.path_forward}")
            return list(link.path_forward)

        # 1) No cache → build an A* state from our current *real* pose
        real = self.agent_current_pose 
        if real is None:
            raise RuntimeError("No last_real_pose available for A* start!")
        sx, sy, sd = real
        start = State(int(round(sx)), int(round(sy)), int(sd))

        # target node’s stored map pose
        gx, gy, gd = emap.get_pose(v)
        goal = State(int(round(gx)), int(round(gy)), int(gd))

        print(f"[get_primitives] no cache {u}->{v}, A* from {start} → {goal}")

        # 2) Run your egocentric‐aware A*:
        egocentric_process = self.key.models_manager.egocentric_process
        prims = self.astar_prims(
            start,
            goal,
            egocentric_process,
            num_samples=5
        )

        print(f"[get_primitives] A* returned for {u}->{v}: {prims}")

        return prims


    def _find_link(self, u: int, v: int):
        """
        Scan your ExperienceMap for a u→v link; return it or None.
        """
        emap = self.models_manager.memory_graph.experience_map
        for exp in emap.exps:
            if exp.id == u:
                for link in exp.links:
                    if link.target.id == v:
                        return link
        return None




    # Cardinal motion vectors for dir ∈ {0=E,1=S,2=W,3=N}
    

    def heuristic(self, s: State, g: State) -> float:
        # Manhattan distance + minimal turn cost
        manh = abs(s.x - g.x) + abs(s.y - g.y)
        turn = min((s.d - g.d) % 4, (g.d - s.d) % 4)
        return manh + turn

    def astar_prims(
    self,
    start: State,
    goal:  State,
    egocentric_process,
    num_samples: int = 5
    ) -> list[str]:
        
        """
        A* in (x,y,dir)-space, but we skip ANY forward candidate
        whose entire prefix fails the egocentric collision check.
        """
        print(f"[A*] start={start}  goal={goal}")
        open_pq = [(self.heuristic(start, goal), 0, start, [])]  # (f, g, state, seq)
        g_score = { start: 0 }
        closed   = set()
        step = 0

        # helper: pack a list of 'forward'/'left'/'right' into a (T,3) float tensor
        def to_onehot_tensor(seq: list[str]) -> torch.Tensor:
            mapping = {'forward': [1,0,0], 'right': [0,1,0], 'left': [0,0,1]}
            return torch.tensor([mapping[a] for a in seq], dtype=torch.float32)

        while open_pq:
            f, g, (x,y,d), seq = heapq.heappop(open_pq)
            print(f"[A*][{step}] POP  state={State(x,y,d)}  g={g}  f={f}  seq={seq!r}")
            step += 1

            if (x,y,d) in closed:
                print("    SKIP (closed)")
                continue
            closed.add((x,y,d))

            # success only if we've reached the right cell and the correct orientation
            if (x, y, d) == (goal.x, goal.y, goal.d):
                print(f"[A*] reached goal at step {step} → seq={seq!r}")
                return seq

            for act in ("left","right","forward"):
                if act == "forward":
                    dx, dy = DIR_VECS[d]
                    nx, ny, nd = x + dx, y + dy, d
                elif act == "left":
                    nx, ny, nd = x, y, (d - 1) % 4
                else:  # right
                    nx, ny, nd = x, y, (d + 1) % 4

                ns = State(nx, ny, nd)
                print(f"    try {act!r} -> next={ns}", end="")

                if (nx,ny,nd) in closed:
                    print("   SKIP (closed)")
                    continue

                new_seq = seq + [act]

                # **egocentric** collision check
                tensor = to_onehot_tensor(new_seq)               # shape (T,3)
                safe   = egocentric_process.egocentric_policy_assessment(
                            tensor, num_samples=num_samples
                        )
                if safe.shape[0] < tensor.shape[0]:
                    print("   SKIP (ego-collision)")
                    continue

                # accept → push into open
                g2 = g + 1
                h2 = self.heuristic(ns, goal)
                f2 = g2 + h2
                old = g_score.get(ns, float("inf"))
                if g2 < old:
                    g_score[ns] = g2
                    print(f"   PUSH g={g2} h={h2} f={f2} seq={new_seq!r}")
                    heapq.heappush(open_pq, (f2, g2, ns, new_seq))
                else:
                    print(f"   SKIP (worse g: {g2} ≥ {old})")

        print("[A*] no path found → returning empty")
        return []
