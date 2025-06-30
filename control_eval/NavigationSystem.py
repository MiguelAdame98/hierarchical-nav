
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
from nltk.grammar import Nonterminal, Production
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

        self._pose_hist: deque[tuple[float,float]] = deque(maxlen=8)
        
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
    
    def _weighted_prod_choice(self, prods):
        if random.random() < 0.8:                 # ε = 0.2 exploration
            return max(prods, key=lambda p: p.prob())
        weights = [p.prob() for p in prods]
        return random.choices(prods, weights=weights, k=1)[0]

    # ─────────────────────────────────────────────────────────────
    # Replacement for generate_plan_with_pcfg
    # ─────────────────────────────────────────────────────────────
    def generate_plan_with_pcfg(self, grammar: PCFG) -> list[tuple[int, int]]:
        """
        Probabilistic sampler that respects the PCFG production weights.

        1. Recursively expand from start symbol 'NAVPLAN'
        2. Collect terminal tokens ('STEP_u_v')
        3. Tokenise into edge tuples, reset indices and progress counters
        """

        agenda = [Nonterminal("NAVPLAN")]
        sentence: list[str] = []

        # --- 1. stochastic top-down expansion -------------
        while agenda:
            sym = agenda.pop()            # depth-first is fine
            if isinstance(sym, Nonterminal):
                prods = grammar.productions(lhs=sym)
                if not prods:
                    # non-expanded nonterminal – abort
                    raise ValueError(f"No production for {sym}")
                prod = self._weighted_prod_choice(prods)
                rhs = list(reversed(prod.rhs()))   # push in reverse for L-to-R
                agenda.extend(rhs)
            else:
                # terminal string
                sentence.append(str(sym))

            # safety guard
            if len(sentence) > 200:
                raise RuntimeError("PCFG expansion runaway (>200 terminals)")

        plan_str = " ".join(sentence)
        # --- 2. tokenise STEP_u_v → (u,v) ----------------
        self.full_plan_tokens = self._tokenize_plan(plan_str)
        self.token_idx = self.prim_idx = 0
        self.token_prims.clear()
        self.plan_progress = 0.0

        # --- 3. store final target node ID ---------------
        if self.full_plan_tokens:
            _, self.target_node_id = self.full_plan_tokens[-1]
        else:
            self.target_node_id = None

        return self.full_plan_tokens
        
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
    def push_pose(self, pose):
        """pose = (x,y,θ) or dict with x,y"""
        if pose is None:
            return
        if isinstance(pose, dict):
            self._pose_hist.append((float(pose["x"]), float(pose["y"])))
        else:
            self._pose_hist.append((float(pose[0]), float(pose[1])))

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
    
    
    
# Observer integration (to be added to your step function)
        # --------------------------------------------------------------
    # Navigation quality metric  (0 – 1)
    # --------------------------------------------------------------
    def navigation_grade(self) -> float:
        """
        1.0  → advancing smoothly along plan
        0.0  → no plan or completely stalled
        """
        # --- 1. Plan progress component -------------------------
        prog = self.plan_progress            # -1, 0-1, 1.1
        if prog < 0 or prog > 1.05:          # no plan or finished
            plan_part = 0.0
        else:
            plan_part = prog                 # already 0..1

        # --- 2. Movement component ------------------------------
        if len(self._pose_hist) < 4:
            move_part = 0.5                  # neutral until we have data
        else:
            # average step length over last N-1 segments
            segs = [
                math.hypot(x2 - x1, y2 - y1)
                for (x1, y1), (x2, y2) in zip(self._pose_hist, list(self._pose_hist)[1:])
            ]
            avg_step = sum(segs) / len(segs)
            # assume 0.30 cell-width per healthy step
            move_part = max(0.0, min(1.0, avg_step / 0.30))

        # --- 3. Combine (weighted) ------------------------------
        return round(0.6 * plan_part + 0.4 * move_part, 3)

    
    def build_pcfg_from_memory(self) -> PCFG:
        mg    = self.memory_graph
        emap  = mg.experience_map
        start = mg.get_current_exp_id()

        # 1. adjacency list
        graph = {e.id: [l.target.id for l in e.links] for e in emap.exps}

        # 2. confidence-weighted goal prior
        goals, total_w = {}, 0.0
        sx, sy, _ = emap.get_pose(start)
        for exp in emap.exps:
            if exp.id == start:
                continue
            conf = getattr(exp, "confidence", 1.0)
            dist = math.hypot(exp.x_m - sx, exp.y_m - sy) + 1e-5
            w = conf / dist
            goals[exp.id] = w
            total_w += w

        # 3. small BFS to list a few paths per goal
        def paths(src, dst, k=12, depth=15):
            out, q = [], deque([[src]])
            while q and len(out) < k:
                p = q.popleft()
                if p[-1] == dst:
                    out.append(p)
                elif len(p) < depth:
                    for nb in graph.get(p[-1], []):
                        if nb not in p:
                            q.append(p + [nb])
            return out

        rules = defaultdict(list)
        for tgt, w in goals.items():
            rules["NAVPLAN"].append((f"PATH_{tgt}", w / total_w))
            for path in paths(start, tgt):
                rhs = " ".join(f"STEP_{u}_{v}" for u, v in zip(path, path[1:]))
                rules[f"PATH_{tgt}"].append((rhs, 1.0))  # equal weight per path

        # 4. make every STEP symbol a terminal
        for lhs in list(rules):
            if lhs.startswith("PATH_"):
                for rhs, _ in rules[lhs]:
                    for tok in rhs.split():
                        if tok not in rules:           # first time we see it
                            u, v = tok.split("_")[1:3]
                            rules[tok].append((f"'step({u},{v})'", 1.0))

        # 5. serialise
        lines = []
        for lhs, prods in rules.items():
            Z = sum(p for _, p in prods)
            for rhs, p in prods:
                lines.append(f"{lhs} -> {rhs} [{p/Z:.4f}]")
        return PCFG.fromstring("\n".join(lines))
          
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
        real = self.get_current_pose()
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
        emap = self.memory_graph.experience_map
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
# -----------------------------------------------------------
# 0.  Minimal stubs for MemoryGraph / ExperienceMap
# -----------------------------------------------------------
class DummyLink:
    def __init__(self, target):  self.target = target

class DummyExp:
    def __init__(self, _id, x, y, conf=1.0):
        self.id = _id
        self.x_m, self.y_m = x, y
        self.confidence = conf
        self.links = []

class DummyExperienceMap:
    def __init__(self, exps):
        self.exps = exps
    def get_pose(self, eid):
        e = next(e for e in self.exps if e.id == eid)
        return (e.x_m, e.y_m, 0)

class DummyMemoryGraph:
    def __init__(self, exps):
        self.experience_map = DummyExperienceMap(exps)
        self._current = 0
    def get_current_exp_id(self):
        return self._current
    def set_current(self, eid):
        self._current = eid

# -----------------------------------------------------------
# 1.  Build a tiny map: 0 ↔ 1 ↔ 2 with confidences
# -----------------------------------------------------------
n0, n1, n2 = DummyExp(0, 0, 0), DummyExp(1, 1, 0), DummyExp(2, 2, 0)
n0.links.append(DummyLink(n1))
n1.links.append(DummyLink(n0)); n1.links.append(DummyLink(n2))
n2.links.append(DummyLink(n1))
mem_graph = DummyMemoryGraph([n0, n1, n2])

# -----------------------------------------------------------
# 2.  Live pose variable + lambda getter
# -----------------------------------------------------------
current_pose = [0.0, 0.0, 0]          # mutable list so lambda sees updates
pose_lambda  = lambda: current_pose

# -----------------------------------------------------------
# 3.  Instantiate NavigationSystem
# -----------------------------------------------------------
nav = NavigationSystem(mem_graph, pose_lambda)

# Monkey-patch get_primitives (edge → two fake primitives)
nav.get_primitives = lambda u, v: [f"edge({u}->{v})-a", f"edge({u}->{v})-b"]
# Monkey-patch astar_prims (evade detour)
nav.astar_prims   = lambda *a, **kw: ["detour-left", "detour-fwd"]

# -----------------------------------------------------------
# 4.  Build PCFG & initial plan
# -----------------------------------------------------------
grammar = nav.build_pcfg_from_memory()
plan    = nav.generate_plan_with_pcfg(grammar)
print("\n--- Sampled plan:", plan, "\n")

# -----------------------------------------------------------
# 5.  Helper to print nav state each tick
# -----------------------------------------------------------
def dump(prefix):
    print(f"{prefix}  token_idx={nav.token_idx}  prim_idx={nav.prim_idx} "
          f"progress={nav.plan_progress:.3f}  pose_hist={list(nav._pose_hist)}")

# -----------------------------------------------------------
# 6.  Simulate ticks
# -----------------------------------------------------------
print(">> plan_following for 4 primitives")
for _ in range(4):
    prims, _ = nav.universal_navigation("plan_following")
    current_pose[0] += 0.3            # pretend we moved forward
    nav.push_pose(tuple(current_pose))
    dump(f"  issued {prims}")

print("\n>> switch to evade for 1 tick")
prims, _ = nav.universal_navigation("evade")
current_pose[0] += 0.1
nav.push_pose(tuple(current_pose))
dump(f"  evade {prims}")

print("\n>> back to plan_following for 3 primitives")
for _ in range(3):
    prims, _ = nav.universal_navigation("plan_following")
    current_pose[0] += 0.3
    nav.push_pose(tuple(current_pose))
    dump(f"  issued {prims}")

print("\n>> force replan (simulate blocked edge)")
prims, _ = nav.universal_navigation("replan")
dump(f"  after replan {prims}")

print("\n>> finish new plan")
while True:
    prims, _ = nav.universal_navigation("plan_following")
    if not prims:
        break
    current_pose[0] += 0.3
    nav.push_pose(tuple(current_pose))
    dump(f"  issued {prims}")

print("\n### Final plan_progress =", nav.plan_progress,
      "navigation_grade =", nav.navigation_grade())
