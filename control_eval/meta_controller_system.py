#meta_controller_system.py

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
# ---------------------------------------------------
# Utility placeholders (replace with real implementations)
# ---------------------------------------------------

def no_vel_no_action(obs):
    """Strip velocity/action from raw observation."""
    return obs

# A generic PCFG sampler stub (implement depth-limited top-down sampling)
def sample_from_pcfg(grammar: PCFG, symbol=None, max_depth=50):
    """
    A simple top-down stochastic PCFG sampler.
    """
    if symbol is None:
        symbol = grammar.start()

    def expand(sym, depth):
        # Terminal: no .symbol() method or depth exhausted
        if depth <= 0 or not hasattr(sym, 'symbol'):
            return [str(sym)]
        prods = grammar.productions(lhs=sym)
        probs = [p.prob() for p in prods]
        total = sum(probs)
        if total <= 0:
            return [str(sym)]
        weights = [p/total for p in probs]
        prod = random.choices(prods, weights=weights, k=1)[0]
        result = []
        for rhs_sym in prod.rhs():
            result += expand(rhs_sym, depth-1)
        return result

    return expand(symbol, max_depth)

# -------------------------------------
# Hierarchical Dynamic PCFG Builder
# -------------------------------------
class HierarchicalDynamicPCFG:
    """
    Three-mode PCFG builder: EXPLORE, NAVIGATE, RECOVER.
    Builds, evolves, and returns the active grammar.
    """
    MODES = ('EXPLORE', 'NAVIGATE', 'RECOVER')

    def __init__(self, key, pattern_detector=None, stats_collector=None):
        # key: the agent instance with models_manager, etc.
        self.memory_graph      = self.key.models_manager.memory_graph
        self.pattern_detector  = pattern_detector
        self.stats_collector   = stats_collector
        self.current_mode      = 'EXPLORE'
        self.level_grammars    = {}
        self.history           = []
        self.mg      = self.key.models_manager.memory_graph
        self.emap=self.mg.experience_map

    def set_mode(self, mode: str):
        assert mode in self.MODES, f"Unknown mode {mode}"
        self.current_mode = mode

    def get_grammar(self) -> PCFG:
        if self.current_mode not in self.level_grammars:
            # build on first request
            if self.current_mode == 'NAVIGATE':
                self.level_grammars['NAVIGATE'] = self.build_pcfg_from_memory()
            else:
                self.level_grammars[self.current_mode] = self.build_for_mode(self.current_mode)
        return self.level_grammars[self.current_mode]

    def build_for_mode(self, mode: str) -> PCFG:
        if mode == 'EXPLORE':
            return self._build_exploration_grammar()
        if mode == 'NAVIGATE':
            return self.build_pcfg_from_memory()
        if mode == 'RECOVER':
            return self._build_recover_grammar()
        raise ValueError(f"Unsupported mode: {mode}")

    def evolve(self, feedback: dict):
        # record event
        self.history.append((self.current_mode, time.time(), feedback))
        # update detectors/stats
        if self.stats_collector:
            self.stats_collector.update(feedback)
        if self.pattern_detector and 'sequence' in feedback:
            self.pattern_detector.update(feedback['sequence'])
        # rebuild
        self.level_grammars[self.current_mode] = self.build_for_mode(self.current_mode)

    def _build_exploration_grammar(self) -> PCFG:
        rules = []
        # Example: simple curiosity vs gap-fill
        rules.append("EXPLORE -> RANDOM_WALK [0.5]")
        rules.append("EXPLORE -> GAP_FILL [0.5]")
        rules.append("RANDOM_WALK -> 'forward' [1.0]")
        rules.append("GAP_FILL -> 'left' [0.5] 'forward' [0.5]")
        return PCFG.fromstring("\n".join(rules))

    def _build_recover_grammar(self) -> PCFG:
        rules = []
        # Example recover primitives
        rules.append("RECOVER -> RELOCALIZE [0.6]")
        rules.append("RECOVER -> BACKTRACK [0.4]")
        rules.append("RELOCALIZE -> 'scan' [1.0]")
        rules.append("BACKTRACK -> 'left' [0.5] 'forward' [0.5]")
        return PCFG.fromstring("\n".join(rules))
    

    def build_pcfg_from_memory(self):
        emap    = self.emap
        current = self.mg.get_current_exp_id()
        exps_by_dist = self.mg.get_exps_organised(current)
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
        emap = self.emap

        # 0) do we already have a stored primitive path?
        link = self._find_link(u, v)
        if link and link.path_forward:
            print(f"[get_primitives] cached path for {u}->{v}: {link.path_forward}")
            return list(link.path_forward)

        # 1) No cache → build an A* state from our current *real* pose
        real = self.key.agent_current_pose 
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
            egocentric_process=egocentric_process,
            num_samples=5
        )

        print(f"[get_primitives] A* returned for {u}->{v}: {prims}")

        return prims


    def _find_link(self, u: int, v: int):
        """
        Scan your ExperienceMap for a u→v link; return it or None.
        """
        emap = self.emap
        for exp in emap.exps:
            if exp.id == u:
                for link in exp.links:
                    if link.target.id == v:
                        return link
        return None
    
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
    
    def print_adaptive_pcfg_plans(self,grammar, max_trials=5, initial_depth=4, max_depth_limit=25):
        print("[DEBUG] Attempting to generate plans adaptively...")

        depth = initial_depth
        while depth <= max_depth_limit:
            try:
                plans = list(generate(grammar, depth=depth))
                if plans:
                    print(f"[PLAN DEBUG] {len(plans)} plans found at depth {depth}:")
                    for i, plan in enumerate(plans):
                        print(f"  Plan {i+1}: {' '.join(plan)}")
                    #return plans  # Optionally return for use
                else:
                    print(f"[PLAN DEBUG] No plans at depth {depth}. Increasing depth...")
            except Exception as e:
                print(f"[PLAN DEBUG] Generation failed at depth {depth}: {e}")
            depth += 2  # Increase search depth gradually

        print("[PLAN DEBUG] No valid plans found up to max depth limit.")
        return []
    def replan(self):
        """Rebuild & adapt PCFG, sample new plan via self.key context."""
        base = self.build_pcfg_from_memory()
        ap = AdaptivePCFG(base)
        ap.apply_meta(self)
        grammar = ap.to_nltk_grammar(start=base.start())
        self.key.current_plan = sample_from_pcfg(grammar)
        print(f"REPLAN: new plan length={len(self.key.current_plan)}")

# -----------------------------
# Bayesian Progress Observer
# -----------------------------
class BayesianProgressObserver:
    """
    Belief state over modes: EXPLORE, NAVIGATE, RECOVER.
    """
    def __init__(self):
        self.beliefs = {'EXPLORE':0.33,'NAVIGATE':0.33,'RECOVER':0.34}

    def _likelihood(self, mode: str, fb: dict) -> float:
        # Define simple likelihoods
        if mode == 'EXPLORE':   return 0.9 if fb.get('info_gain',0)<0.5 else 0.1
        if mode == 'NAVIGATE':  return 0.9 if fb.get('progress',0)>0.2 else 0.2
        if mode == 'RECOVER':   return 0.9 if fb.get('agent_lost',False) else 0.1
        return 0.1

    def observe(self, feedback: dict) -> str:
        # Bayes update
        post = {}
        total = 0.0
        for m,prior in self.beliefs.items():
            like = self._likelihood(m, feedback)
            post[m] = prior * like
            total += post[m]
        if total>0:
            for m in post: post[m] /= total
        self.beliefs = post
        # return mode with max posterior
        return max(post, key=post.get)

# -----------------------------
# MetaController
# -----------------------------
class MetaController:
    """
    Supervises EXPLORE, NAVIGATE, RECOVER using PCFG + Bayes.
    """
    def __init__(self, memory_graph):
        self.pcfg_builder = HierarchicalDynamicPCFG(memory_graph)
        self.observer     = BayesianProgressObserver()
        self.current_plan = []

    def step(self, feedback: dict):
        # 1) Evolve PCFG
        self.pcfg_builder.evolve(feedback)
        # 2) Bayes to pick mode
        new_mode = self.observer.observe(feedback)
        if new_mode != self.pcfg_builder.current_mode:
            self.pcfg_builder.set_mode(new_mode)
        # 3) Sample plan
        grammar = self.pcfg_builder.get_grammar()
        self.current_plan = sample_from_pcfg(grammar)
        return self.current_plan


    # ---------------------------------------------------
    # Integration & Helpers (use self.key)
    # ---------------------------------------------------

    def measure_progress(self, prev, curr):
        if prev is None:
            return 0.0
        return min(1.0, np.linalg.norm(np.array(curr)-np.array(prev))/0.5)

    def handle_meta_action(self, action):
        """Dispatch actions on the attached `self.key` agent."""
        if self.key is None:
            raise RuntimeError("MetaController.key not attached")
        if action == 'SWITCH_TO_LOCAL_RECOVERY':
            self.key.start_astar_recovery()
        elif action in ('REPLAN_WITHOUT_CURRENT_NODE','EXECUTE_NEW_PLAN','REPLAN_WITH_NEW_INFORMATION'):
            self.replan()
        elif action == 'SWITCH_TO_EXPLORATION':
            self.key.apply_exploration()

    
# ---------------------------------------------------
class AdaptivePCFG:
    """
    Wraps a PCFG and applies meta-controller feedback.
    """
    def __init__(self, base_grammar: PCFG):
        self.rules = list(base_grammar.productions())
        self.excluded = set()
        self.adjustments = {}

    def exclude_node(self, node):
        self.excluded.add(node)

    def adjust_probability(self, node, factor):
        self.adjustments[node] = factor

    def normalize(self):
        groups = defaultdict(list)
        for r in self.rules:
            groups[r.lhs()].append(r)
        for lhs, rs in groups.items():
            total = sum(r.prob() for r in rs)
            if total > 0:
                for r in rs:
                    r._prob = r.prob()/total

    def apply_meta(self, meta: MetaController):
        # mark excluded nodes
        for n in meta.get_excluded_nodes():
            self.excluded.add(n)
        # store confidence factors
        for n, c in meta.get_node_confidence().items():
            self.adjustments[n] = c
        # adjust rule probs
        for r in self.rules:
            rep = str(r)
            if any(n in rep for n in self.excluded):
                r._prob = 0.0
            else:
                for n, c in self.adjustments.items():
                    if n in rep:
                        r._prob = r.prob()*c
        # normalize
        groups = defaultdict(list)
        for r in self.rules:
            groups[r.lhs()].append(r)
        for rs in groups.values():
            tot = sum(r.prob() for r in rs)
            if tot>0:
                for r in rs: r._prob = r.prob()/tot

    def to_nltk(self, start):
        lines = [f"{r.lhs()} -> {' '.join(map(str,r.rhs()))} [{r.prob():.6f}]"
                 for r in self.rules]
        return PCFG.fromstring("\n".join(lines), start=start)


    


# ---------------------------------------------------
# Agent Integration (MetaController Only)
# ---------------------------------------------------
class Agent:
    def __init__(self, env, models_manager, explorative_behaviour):
        self.env = env
        self.models_manager = models_manager
        self.explorative_behaviour = explorative_behaviour
        self.meta = MetaController()
        self.current_plan = []
        self.plan_idx = 0
        self.last_pose = None

    def agent_step(self, action):
        # Store previous pose for progress measurement
        prev = self.last_pose
        # Execute action
        obs, _, _, _ = self.env.step(action)
        obs = no_vel_no_action(obs)
        pos = obs['pose']
        self.last_pose = pos

        # Update environment models
        self.models_manager.digest(obs)
        info = self.models_manager.get_best_place_hypothesis()['info_gain']
        node = self.agent_situate_memory()

        # Measure progress
        prog = self.measure_progress(prev, pos)

        # Update MetaController
        meta_action = self.meta.update(
            plan_prog=prog,
            agent_lost=self.models_manager.agent_lost(),
            info_gain=info,
            current_node=node,
            current_pose=pos
        )

        # Handle any meta action
        if meta_action != 'CONTINUE':
            self.handle_meta_action(meta_action)

        return self.models_manager.agent_lost(), obs

    def measure_progress(self, prev, curr):
        """Compute normalized progress based on movement."""
        if prev is None:
            return 0.0
        dist = np.linalg.norm(np.array(curr) - np.array(prev))
        return min(1.0, dist / 0.5)

    def handle_meta_action(self, action):
        if action == 'SWITCH_TO_LOCAL_RECOVERY':
            self.start_astar_recovery()
        elif action.startswith('REPLAN'):
            self.replan()
        elif action == 'SWITCH_TO_EXPLORATION':
            self.apply_exploration()
        # 'CONTINUE' does nothing

    def apply_exploration(self):
        """
        Exploration stub updated to integrate with MetaController.
        Future extension: adapt exploration based on meta.current_state.
        """
        print('Exploration mode triggered by MetaController')
        return [], 0

class NavigationSystem:
    def __init__(self, memory_graph, get_primitives_func, astar_prims_func, get_current_pose_func):
        self.memory_graph = memory_graph
        self.get_primitives = get_primitives_func
        self.astar_prims = astar_prims_func
        self.get_current_pose = get_current_pose_func
        
        # Navigation state
        self.current_full_plan = None
        self.current_plan_tokens = []
        self.current_token_index = 0
        self.target_node_id = None
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
    
    def generate_plan_with_pcfg(self, grammar):
        """Generate a new plan using PCFG sampling"""
        # Sample from PCFG (you already have this logic)
        # This would use your existing build_pcfg_from_memory logic
        sampled_plan = self._sample_from_pcfg(grammar)
        
        # Parse the plan to extract target node
        self.target_node_id = self._extract_target_from_plan(sampled_plan)
        self.current_full_plan = sampled_plan
        
        # Tokenize the plan
        self.current_plan_tokens = self._tokenize_plan(sampled_plan)
        self.current_token_index = 0
        
        return sampled_plan
    
    def _tokenize_plan(self, full_plan):
        """Break full plan into individual step tokens"""
        # Example: "STEP_9_9 STEP_9_8 STEP_8_5 STEP_5_4 STEP_4_6"
        # Returns: ["STEP_9_9", "STEP_9_8", "STEP_8_5", "STEP_5_4", "STEP_4_6"]
        if isinstance(full_plan, str):
            tokens = full_plan.strip().split()
            return [token for token in tokens if token.startswith('STEP_')]
        return []
    
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
    
    def universal_navigation_function(self, current_pose, objective, submode="plan_following"):
        """
        Universal function that handles all navigation submodes
        
        Args:
            current_pose: Current agent position
            objective: Target (can be pose or node_id)
            submode: "plan_following", "evade", or "replan"
            
        Returns:
            (primitives, n_actions)
        """
        if submode == "plan_following":
            return self._handle_plan_following(current_pose, objective)
            
        elif submode == "evade":
            return self._handle_evade(current_pose, objective)
            
        elif submode == "replan":
            return self._handle_replan(current_pose, objective)
            
        else:
            raise ValueError(f"Unknown submode: {submode}")
    
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


# Updated apply_exploration method
def apply_exploration(self) -> tuple[list, int]:
    """
    Updated exploration method with navigation system integration
    """
    # Get current HMM state
    mode, submode = (
        getattr(self, "prev_mode", None) or "EXPLORE",
        getattr(self, "prev_submode", None) or "ego_allo",
    )
    print(f"[HMM] mode={mode}  submode={submode}")

    # Handle other modes (RECOVER, TASK_SOLVING, EXPLORE) as before...
    # [Previous code remains the same]

    # NAVIGATE mode with new implementation
    if mode == "NAVIGATE":
        # Initialize navigation system
        nav_system = NavigationSystem(
            self.models_manager.memory_graph,
            self.get_primitives,
            self.astar_prims,
            lambda: self.agent_current_pose
        )
        
        # Early feasibility check
        if not nav_system.check_navigation_feasibility():
            print("Navigation not feasible - returning flag for HMM")
            # Set flags for observer to detect and trigger mode change
            self.navigation_flags = nav_system.navigation_flags
            return [], []  # This should trigger mode change via observer
        
        # Check if we need to generate a new plan
        if nav_system.current_full_plan is None:
            grammar = self.build_pcfg_from_memory()
            nav_system.generate_plan_with_pcfg(grammar)
        
        # Handle submodes
        current_pose = self.agent_current_pose
        
        if submode == "plan_following":
            print("Navigate ➜ follow current plan")
            primitives, n_actions = nav_system.universal_navigation_function(
                current_pose, nav_system.target_node_id, "plan_following"
            )
            return primitives, n_actions

        elif submode == "evade":
            print("Evade obstacle")
            primitives, n_actions = nav_system.universal_navigation_function(
                current_pose, nav_system.target_node_id, "evade"
            )
            return primitives, n_actions

        elif submode == "replan":
            print("Objective unreachable lets replan")
            primitives, n_actions = nav_system.universal_navigation_function(
                current_pose, nav_system.target_node_id, "replan"
            )
            return primitives, n_actions

    # [Rest of the method remains the same for other modes]
    return [], []


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


