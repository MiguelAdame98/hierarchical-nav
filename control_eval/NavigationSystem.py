
import time
import random
import numpy as np
from collections import deque
from collections import defaultdict
from nltk import PCFG
import math
import re
import heapq
import torch
from collections import namedtuple
from nltk.parse.generate import generate
from nltk.grammar import Nonterminal, Production
# --- optional dreamer_mg import (safe for test harness) ---
try:
    from dreamer_mg.world_model_utils import DictResizeObs, WMPlanner  # type: ignore
except Exception:
    # Minimal stubs so type hints & calls don't explode during testing
    class WMPlanner:  # pragma: no cover
        def astar_prims(self, *args, **kwargs):
            return []
    class DictResizeObs:  # pragma: no cover
        pass

State = namedtuple("State", ["x","y","d"])
DIR_VECS = [(1,0),(0,1),(-1,0),(0,-1)]
ACTIONS   = ["forward","left","right"]

class NavigationSystem:
    def __init__(self, memory_graph, get_current_pose_func, planner: WMPlanner | None = None  ):
        self.memory_graph      = memory_graph          # live reference
        self.get_current_pose  = get_current_pose_func # lambda ↦ (x,y,d)
        
        self.planner = planner 
        # ─ plan bookkeeping ─
        self.full_plan_tokens: list[tuple[int, int]] = []   # [(u,v), …]
        self.token_idx   = 0        # which edge
        self.token_prims: list[str] = []
        self.prim_idx    = 0
        self.plan_progress = -1.0   # -1=no plan, 0-1 executing, 1.1 done
        # ─ misc state you already had ─
        self.target_node_id   = None
        self.current_mode= None
        self.navigation_flags = {}
        self._prev_token_idx = -2
        self._prev_prim_idx  = -2
        self._stall_count    = 0         
        self._stall_limit    = 5      
        self._evade_injected = False
        self._evade_ticks_since_recalc = 0
        self._evade_recalc_every = 4  

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
        if len(viable_nodes) < 2:
            self.navigation_flags['insufficient_nodes'] = True
            return False
            
        if len(viable_nodes) == 0:
            self.navigation_flags['no_viable_nodes'] = True
            return False
            
        # Check if we have a current plan that's still valid
        if self.full_plan_tokens and self.target_node_id:
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
    @staticmethod
    def to_onehot_list(act: str) -> list[int]:
        """'left'/'right'/'forward' → [0,0,1] / [0,1,0] / [1,0,0]"""
        mapping = {'forward': [1,0,0], 'right': [0,1,0], 'left': [0,0,1]}
        return mapping[act]
    
    def _pcfg_shortest_path(self, graph: dict[int, list[int]], start: int, goal: int):
        """Return one shortest path [start,...,goal] in edge-count metric, or None."""
        if start == goal:
            return [start]
        parent = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in graph.get(u, ()):
                if v in parent:
                    continue
                parent[v] = u
                if v == goal:
                    # reconstruct
                    path = [v]
                    while u is not None:
                        path.append(u); u = parent[u]
                    return list(reversed(path))
                q.append(v)
        return None
    
    def build_pcfg_from_memory(self, debug: bool = True) -> PCFG:
        """
        Build a PCFG that *deterministically* targets a single GOAL node chosen by
        a human-like 'sparse-zone' heuristic, gates links by confidence (1/0),
        permanently blacklists zero-confidence pairs, and *proposes* new links
        between nearby, currently unconnected nodes (distance threshold).
        The rest of the grammar structure is preserved.

        Side-effects:
        - self.link_blacklist: Set[frozenset({u,v})] is created/updated.
        - self.goal_node_id: the deterministically chosen goal node.
        - self.graph_used_for_pcfg: adjacency (dict[int, list[int]]) actually used.

        Compatibility:
        - generate_plan_with_pcfg() keeps working: we still start from 'NAVPLAN'
            and emit 'STEP_u_v' terminals; target is the last token's 'v'.

        Notes:
        - Node confidences are **ignored**.
        - Link confidences gate edges: getattr(link, "confidence", 1.0) ∈ {1,0}.
        """
        mg   = self.memory_graph
        emap = mg.experience_map
        start = mg.get_current_exp_id()

        # ---------- A) ensure permanent blacklist ---------------------------------
        if not hasattr(self, "link_blacklist") or self.link_blacklist is None:
            self.link_blacklist = set()   # set[frozenset({u,v})]

        # ---------- B) collect basic node list ------------------------------------
        exps = list(getattr(emap, "exps", []))
        if len(exps) == 0:
            raise RuntimeError("No experiences in experience_map")
        id2exp = {e.id: e for e in exps}

        # ---------- C) adjacency from CONFIDENT links + blacklist ------------------
        graph = self._pcfg_build_adjacency_with_confidence_and_inferred(
            exps=exps,
            start_id=start,
            blacklist=self.link_blacklist,
            # Heuristic default for MiniGrid-like spacing (~3.0 units per grid)
            infer_link_max_dist=float(getattr(self, "infer_link_max_dist", 3.4)),
            debug=debug
        )

        # Persist for inspection
        self.graph_used_for_pcfg = graph

        if debug:
            print("[PCFG DEBUG] adjacency (post-confidence+inferred):", {k: sorted(v) for k,v in graph.items()})

        # ---------- D) choose GOAL deterministically via sparse-zone heuristic -----
        goal = self._pcfg_select_goal_node_via_zones(
            exps=exps, start_id=start, graph=graph, debug=debug
        )
        self.goal_node_id = goal

        # If still None for any reason, fallback to farthest reachable node
        if goal is None:
            if debug:
                print("[PCFG DEBUG] Goal via zones unavailable → fallback to farthest reachable node.")
            dists = self._pcfg_bfs_all_dists(graph, start)
            if len(dists) > 0:
                far_node = max(dists.items(), key=lambda kv: (kv[1], kv[0]))[0]
                goal = far_node
            else:
                goal = start
            self.goal_node_id = goal

        # As an additional safeguard, if no path exists to 'goal', fallback.
        if not self._pcfg_is_reachable(graph, start, goal):
            if debug:
                print(f"[PCFG DEBUG] Goal {goal} unreachable from {start} → fallback to farthest reachable.")
            dists = self._pcfg_bfs_all_dists(graph, start)
            if len(dists) > 0:
                far_node = max(dists.items(), key=lambda kv: (kv[1], kv[0]))[0]
                goal = far_node
            else:
                goal = start
            self.goal_node_id = goal

        if debug:
            print(f"[PCFG DEBUG] Chosen GOAL node: {goal} (start={start})")

        # ---------- E) enumerate a few path candidates to GOAL ---------------------
        def all_paths(src, dst, k=16, depth=50):
            out, q = [], deque([[src]])
            seen_paths = set()
            while q and len(out) < k:
                p = q.popleft()
                u = p[-1]
                if u == dst:
                    tup = tuple(p)
                    if tup not in seen_paths:
                        out.append(p); seen_paths.add(tup)
                    continue
                if len(p) >= depth:
                    continue
                for nb in graph.get(u, ()):   # deterministic by sorted adj
                    if nb not in p:
                        q.append(p + [nb])
            return out

        # 1) Always try to get the shortest path first
        sp = self._pcfg_shortest_path(graph, start, goal)
        if debug:
            sp_len = (len(sp) - 1) if sp else None
            print(f"[PCFG DEBUG] shortest path edges: {sp_len}")

        # 2) Choose a depth cap adaptively
        #    Allow an instance override: self.pcfg_depth_cap, self.pcfg_k_paths, self.pcfg_depth_margin
        k_paths   = int(getattr(self, "pcfg_k_paths", 16))
        margin    = int(getattr(self, "pcfg_depth_margin", 4))
        if getattr(self, "pcfg_depth_cap", None):
            depth_cap = int(self.pcfg_depth_cap)
        else:
            if sp is not None:
                depth_cap = (len(sp) - 1) + max(margin, 1)
            else:
                depth_cap = len(exps) + max(margin, 1)   # safe upper bound on simple paths

        # 3) Collect paths: shortest (if any) + additional alternatives (deduped)
        paths = []
        if sp:
            paths.append(sp)
        extras = all_paths(start, goal, k=k_paths, depth=depth_cap)
        for p in extras:
            if not sp or p != sp:
                paths.append(p)

        if debug:
            print(f"[PCFG DEBUG] depth_cap={depth_cap}  total_paths={len(paths)}")

        # If still no path (unreachable), synthesize trivial one
        if not paths:
            paths = [[start]]
        # ---------- F) construct production rules (keep structure) -----------------
        rules = defaultdict(list)

        # NAVPLAN → PATH_goal (deterministic single option)
        rules["NAVPLAN"].append((f"PATH_{goal}", 1.0))

        # PATH_goal → STEP_*_* ...  (equal weight per enumerated path)
        for path in paths:
            step_tokens = []
            #if not self._at_node_exact(start):
                #step_tokens.append(f"STEP_{start}_{start}")  # self-edge only if away
            step_tokens += [f"STEP_{u}_{v}" for u, v in zip(path, path[1:])]
            rhs = " ".join(step_tokens) if step_tokens else f"STEP_{start}_{start}"
            rules[f"PATH_{goal}"].append((rhs, 1.0))

        # Every STEP_u_v becomes a terminal
        for lhs in list(rules.keys()):
            if lhs.startswith("PATH_"):
                for rhs, _ in rules[lhs]:
                    for tok in rhs.split():
                        if tok not in rules:
                            rules[tok].append((f"'{tok}'", 1.0))

        # ---------- G) serialise to NLTK PCFG -------------------------------------
        lines = []
        for lhs, prods in rules.items():
            Z = sum(p for _, p in prods) or 1.0
            for rhs, p in prods:
                lines.append(f"{lhs} -> {rhs} [{p/Z:.6f}]")

        grammar_src = "\n".join(lines)
        if debug:
            print("[PCFG DEBUG] Final grammar:\n" + grammar_src)

        return PCFG.fromstring(grammar_src)

    def _reset_token_prims(self):
        import torch
        if torch.is_tensor(self.token_prims):
            # keep type stable for downstream code
            self.token_prims = torch.empty((0, 3), dtype=self.token_prims.dtype, device=self.token_prims.device)
        else:
            self.token_prims = []
    # ──────────────────────────────────────────────────────────────────────────────
    # Helper methods (paste inside the same class)
    # ──────────────────────────────────────────────────────────────────────────────

    def _pcfg_is_reachable(self, graph: dict[int, list[int]], start: int, goal: int) -> bool:
        if start == goal:
            return True
        q, seen = deque([start]), {start}
        while q:
            u = q.popleft()
            for v in graph.get(u, ()):
                if v == goal:
                    return True
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    def _pcfg_bfs_all_dists(self, graph: dict[int, list[int]], start: int) -> dict[int, int]:
        """BFS in edge-count metric; returns distances to all reachable nodes."""
        d = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in graph.get(u, ()):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def _pcfg_build_adjacency_with_confidence_and_inferred(
        self,
        exps,
        start_id: int,
        blacklist: set,
        infer_link_max_dist: float,
        debug: bool = False
    ) -> dict[int, list[int]]:
        """
        Build a directed adjacency graph from:
        (1) confident links (confidence==1) that are not blacklisted
        (2) inferred links for close-by unconnected nodes (below threshold),
            not blacklisted. Inferred links are added in *both* directions.

        Toggles (attributes on self, default False if absent):
        • pcfg_treat_all_links_confident : if True, treat *all* links as conf=1 and
            do NOT add new pairs to blacklist for this build.
        • pcfg_disable_inferred_links    : if True, skip proximity-based inferred links.
        • pcfg_ignore_blacklist          : if True, ignore blacklist both for existing
            links and inferred links (does *not* mutate the blacklist).
        """
        # Read toggles with safe defaults
        treat_all = bool(getattr(self, "pcfg_treat_all_links_confident", True))
        no_infer  = bool(getattr(self, "pcfg_disable_inferred_links", False))
        ign_bl    = bool(getattr(self, "pcfg_ignore_blacklist", True))

        graph = {e.id: [] for e in exps}
        def canon(u,v): return frozenset((u,v))

        dropped, added_black, added_inferred = 0, 0, 0

        # (1) Existing links with confidence & blacklist gating
        for e in exps:
            for lnk in getattr(e, "links", []):
                # neighbor id robustly
                v = None
                tgt = getattr(lnk, "target", None)
                if tgt is not None:
                    v = getattr(tgt, "id", None)
                if v is None:
                    v = getattr(lnk, "target_id", None)
                if v is None:
                    continue  # malformed link

                pair = canon(e.id, v)
                orig_conf = int(getattr(lnk, "confidence", 1))
                conf = 1 if treat_all else orig_conf

                if conf == 1 and (ign_bl or pair not in blacklist):
                    graph[e.id].append(v)
                else:
                    # Only mutate blacklist when we *aren't* in "treat_all" and *aren't* ignoring blacklist
                    if (orig_conf == 0) and (not treat_all) and (not ign_bl):
                        if pair not in blacklist:
                            blacklist.add(pair)
                            added_black += 1
                    dropped += 1
        # (2) Proximity-inferred links (optional) — **use only (x,y) coordinates**
        if not no_infer:
            emap = self.memory_graph.experience_map

            def _xy_of(e):
                # 1) Prefer explicit JSON-style fields
                x = getattr(e, "x", None)
                y = getattr(e, "y", None)
                if x is None or y is None:
                    # 2) Fallback to emap.get_pose(e.id) and take only (x,y)
                    xx, yy, _ = emap.get_pose(e.id)
                    x, y = xx, yy
                return float(x), float(y)

            pts = {e.id: _xy_of(e) for e in exps}
            ids = sorted(pts.keys())

            # Inference mode & knobs (safe defaults)
            infer_mode   = str(getattr(self, "pcfg_infer_mode", "mutual_knn"))  # 'mutual_knn' | 'knn' | 'radius'
            k_neighbors  = int(getattr(self, "pcfg_infer_k", 2))                 # K for (mutual_)kNN
            mutual_only  = bool(getattr(self, "pcfg_infer_mutual_only", True))   # only add if both u∈kNN(v) and v∈kNN(u)
            max_per_node = int(getattr(self, "pcfg_infer_max_per_node", k_neighbors))
            hard_radius  = float(infer_link_max_dist) if (infer_link_max_dist and infer_link_max_dist > 0) else float("inf")
            dbg_infer    = bool(getattr(self, "pcfg_infer_debug", False))

            # Precompute sorted neighbor candidates by Euclidean distance (x,y only)
            sorted_neighbors: dict[int, list[tuple[float, int]]] = {}
            for u in ids:
                ux, uy = pts[u]
                cand: list[tuple[float, int]] = []
                for v in ids:
                    if v == u:
                        continue
                    pair = canon(u, v)
                    if (not ign_bl) and (pair in blacklist):
                        continue
                    # Skip if already connected either way
                    if (v in graph.get(u, ())) or (u in graph.get(v, ())):
                        continue
                    vx, vy = pts[v]
                    d = math.hypot(ux - vx, uy - vy)
                    cand.append((d, v))
                cand.sort(key=lambda t: t[0])  # nearest first
                sorted_neighbors[u] = cand

            inferred_pairs = set()

            if infer_mode in ("knn", "mutual_knn"):
                # Select up to K neighbors per node under the hard radius
                topk_under_radius: dict[int, list[int]] = {}
                for u in ids:
                    lst = [v for (d, v) in sorted_neighbors[u] if d <= hard_radius]
                    topk_under_radius[u] = lst[:k_neighbors]

                if infer_mode == "knn":
                    # one-sided kNN (still under radius)
                    for u in ids:
                        for v in topk_under_radius[u][:max_per_node]:
                            inferred_pairs.add(canon(u, v))
                else:
                    # mutual-kNN under radius: u↔v only if each is in the other's top-K
                    for u in ids:
                        for v in topk_under_radius[u][:max_per_node]:
                            if u in topk_under_radius.get(v, ()):
                                inferred_pairs.add(canon(u, v))

            elif infer_mode == "radius":
                # Pure radius graph (original behavior, but still only (x,y))
                for i, u in enumerate(ids):
                    ux, uy = pts[u]
                    for v in ids[i+1:]:
                        pair = canon(u, v)
                        if (not ign_bl) and (pair in blacklist):
                            continue
                        if (v in graph.get(u, ())) or (u in graph.get(v, ())):
                            continue
                        vx, vy = pts[v]
                        d = math.hypot(ux - vx, uy - vy)
                        if d <= hard_radius:
                            inferred_pairs.add(pair)

            # Add inferred edges (undirected → add both directions)
            for pair in inferred_pairs:
                u, v = tuple(pair)
                graph[u].append(v)
                graph[v].append(u)

            added_inferred = len(inferred_pairs)

            if debug or dbg_infer:
                # Show a few of the longest inferred edges to catch outliers
                longest = sorted(
                    [(math.hypot(pts[u][0]-pts[v][0], pts[u][1]-pts[v][1]), u, v)
                    for (u, v) in [tuple(p) for p in inferred_pairs]],
                    reverse=True
                )[:5]
                if longest:
                    print("[PCFG DEBUG] inferred (longest first) [dist,u,v]:", longest)
                print(f"[PCFG DEBUG] infer_mode={infer_mode} k={k_neighbors} mutual={mutual_only} "
                    f"hard_radius={hard_radius:.3f} max_per_node={max_per_node}")

        # deterministic adjacency
        for u in graph:
            graph[u] = sorted(set(graph[u]))

        if debug:
            print(f"[PCFG DEBUG] flags: treat_all={treat_all}  no_infer={no_infer}  ignore_blacklist={ign_bl}")
            print(f"[PCFG DEBUG] dropped_links={dropped}  blacklisted_additions={added_black}  inferred_additions={added_inferred}")

        return graph


    def _pcfg_select_goal_node_via_zones(
        self,
        exps,
        start_id: int,
        graph: dict[int, list[int]],
        debug: bool = False
    ) -> int | None:
        """
        Choose a GOAL node by:
        1) Form bounding box of all nodes.
        2) Partition into a 3x3 grid of zones.
        3) Find the zone(s) with the fewest nodes (sparsest).
        4) Choose the node closest to the centroid of the sparsest zone
            (deterministic tie-breaker by node id).
        Fallbacks:
        - If too few nodes (<4) or degenerate bbox, return None.
        """
        nodes = [(e.id, float(getattr(e, "x_m", getattr(e, "x", 0.0))), float(getattr(e, "y_m", getattr(e, "y", 0.0)))) for e in exps]
        if len(nodes) < 4:
            
            return None

        xs = [x for _, x, _ in nodes]
        ys = [y for _, _, y in nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if not (math.isfinite(min_x) and math.isfinite(max_x) and math.isfinite(min_y) and math.isfinite(max_y)):
            return None
        if abs(max_x - min_x) < 1e-6 or abs(max_y - min_y) < 1e-6:
            return None

        # 3x3 grid partition
        Nx = 4
        Ny = 4
        step_x = (max_x - min_x) / Nx
        step_y = (max_y - min_y) / Ny

        # Assign nodes to cells
        cell_counts = [[0 for _ in range(Ny)] for _ in range(Nx)]
        for _, x, y in nodes:
            ix = min(Nx-1, int((x - min_x) / step_x))
            iy = min(Ny-1, int((y - min_y) / step_y))
            cell_counts[ix][iy] += 1

        # Find sparsest cells (min count)
        min_count = min(c for col in cell_counts for c in col)
        sparsest = [(ix, iy) for ix in range(Nx) for iy in range(Ny) if cell_counts[ix][iy] == min_count]

        # Prefer sparsest cell that is farthest from the *start* node position (centroid distance)
        # Get start coordinates
        sx, sy, _ = getattr(self.memory_graph.experience_map, "get_pose")(start_id)
        def cell_centroid(ix, iy):
            cx = min_x + (ix + 0.5) * step_x
            cy = min_y + (iy + 0.5) * step_y
            return cx, cy

        sparsest_sorted = sorted(
            sparsest,
            key=lambda ij: (
                # choose the cell whose centroid is *farthest* from start (to expand frontier)
                -math.hypot(cell_centroid(*ij)[0]-sx, cell_centroid(*ij)[1]-sy),
                ij[0], ij[1]
            )
        )
        best_cell = sparsest_sorted[0] if sparsest_sorted else None
        if best_cell is None:
            return None
        cx, cy = cell_centroid(*best_cell)

        # Choose actual node: nearest to that centroid
        best_node = None
        best_d = float("inf")
        for nid, x, y in nodes:
            d = (x - cx)**2 + (y - cy)**2
            if d < best_d or (d == best_d and (best_node is None or nid < best_node)):
                best_d = d
                best_node = nid

        # Make sure goal is not the current start unless trivial map; if it is, pick farthest node instead
        if best_node == start_id and len(nodes) > 1:
            dists = self._pcfg_bfs_all_dists(graph, start_id)
            if dists:
                best_node = max(dists.items(), key=lambda kv: (kv[1], kv[0]))[0]

        # Finally, avoid unreachable pick (caller still double-checks)
        return best_node
    def generate_plan_with_pcfg(self, grammar: PCFG, debug: bool = True) -> list[tuple[int, int]]:
        """
        Probabilistic sampler that respects the PCFG production weights.

        1. Recursively expand from start symbol 'NAVPLAN'
        2. Collect terminal tokens ('STEP_u_v')
        3. Tokenise into edge tuples, reset indices and progress counters
        """

        agenda = [Nonterminal("NAVPLAN")]
        sentence: list[str] = []

        if debug:
            print("\n[DBG] Starting PCFG expansion from 'NAVPLAN'")
            print(f"[DBG] Initial agenda: {agenda}")

        # --- 1. stochastic top-down expansion -------------
        while agenda:
            sym = agenda.pop()

            if debug:
                print(f"[DBG] Popped symbol: {sym}")

            if isinstance(sym, Nonterminal):
                prods = grammar.productions(lhs=sym)
                if not prods:
                    raise ValueError(f"No production for {sym}")
                prod = self._weighted_prod_choice(prods)
                rhs = list(reversed(prod.rhs()))  # push in reverse for L-to-R
                agenda.extend(rhs)

                if debug:
                    print(f"[DBG] Expanded {sym} using: {prod}")
                    print(f"[DBG] New agenda: {agenda}")
            else:
                sentence.append(str(sym))
                if debug:
                    print(f"[DBG] Added terminal: {sym}")

            # safety guard
            if len(sentence) > 200:
                raise RuntimeError("PCFG expansion runaway (>200 terminals)")

        plan_str = " ".join(sentence)
        if debug:
            print(f"[DBG] Final sentence: {plan_str}")

        # --- 2. tokenise STEP_u_v → (u,v) ----------------
        self.full_plan_tokens = self._tokenize_plan(plan_str)
        self.token_idx = self.prim_idx = 0
        self._reset_token_prims() 
        self.plan_progress = 0.0

        if debug:
            print(f"[DBG] Tokenized plan: {self.full_plan_tokens}")

        # --- 3. store final target node ID ---------------
        if self.full_plan_tokens:
            _, self.target_node_id = self.full_plan_tokens[-1]
        else:
            self.target_node_id = None

        if debug:
            print(f"[DBG] Target node ID: {self.target_node_id}")

        return self.full_plan_tokens
        
    def _tokenize_plan(self, full_plan: str) -> list[tuple[int, int]]:
        """Convert 'STEP_9_8 STEP_8_5 …' → [(9,8), (8,5), …]."""
        return [
            tuple(map(int, t.split('_')[1:3]))
            for t in full_plan.strip().split()
            if t.startswith('STEP_')
        ]
    
    def _load_edge_primitives(self, wm, belief, debug: bool = False):
        """
        Ensure `self.token_prims` is (re)filled for the *current* edge
        identified by `self.token_idx`.

        Policy:
        • If plan done → clear and return.
        • If primitives already queued → leave them (do NOT reload).
        • If EMPTY:
            A) If already at *target* v → LEAVE EMPTY and return
                (caller will skip the edge — avoids start==goal A*).
            B) If at *source* u → fetch edge path via get_primitives(u,v).
            C) Else → inject a detour A* from real pose to v.
        """
        import torch

        # 0) Plan finished?
        if self.token_idx >= len(self.full_plan_tokens):
            self.token_prims = []
            return

        # 1) Already have something queued? leave it alone.
        seq = getattr(self, "token_prims", [])
        if seq is not None:
            if torch.is_tensor(seq):
                if seq.numel() > 0:
                    return
            else:
                if len(seq) > 0:
                    return

        # 2) We only get here if token_prims is EMPTY
        u, v = self.full_plan_tokens[self.token_idx]
        if debug:
            print(f"[_load_edge_prims] need prims for edge {u}->{v}  at_token={self.token_idx}")

        # === SHORT-CIRCUIT: already at the TARGET node v? ===
        # Leave token_prims empty and let step_plan() advance the edge.
        if self._at_node_exact(v):
            if debug:
                print(f"[_load_edge_prims] already at target v={v} → leave empty (caller will skip)")
            self.token_prims = []
            return

        # A) Are we physically at the *source* node u?
        if self._at_node_exact(u):
            if debug: print(f"[_load_edge_prims] docked at u={u} → get_primitives(u,v)")
            prims = self.get_primitives(u, v, wm, belief)
            # normalize; allow empty
            if torch.is_tensor(prims):
                self.token_prims = [] if prims.numel() == 0 else prims
                if debug and prims.numel() == 0:
                    print("[_load_edge_prims] get_primitives returned EMPTY tensor")
            else:
                self.token_prims = prims if prims else []
                if debug and not prims:
                    print("[_load_edge_prims] get_primitives returned EMPTY list")
            return

        # B) Not at source node → inject detour from *real pose* to target v.
        pose = self.get_current_pose()
        if pose is None:
            if debug: print("[_load_edge_prims] no current pose! cannot detour → empty")
            self.token_prims = []
            return

        sx, sy, sd = pose
        gx, gy, gd = self.memory_graph.experience_map.get_pose(v)

        # Extra guard: if start==goal (pose drift/timing), leave empty and let caller skip.
        if int(round(sx)) == int(round(gx)) and int(round(sy)) == int(round(gy)) and int(round(sd)) == int(round(gd)):
            if debug:
                print(f"[_load_edge_prims] start==goal ({sx},{sy},{sd}) → leave empty (caller will skip)")
            self.token_prims = []
            return

        start = State(int(round(sx)), int(round(sy)), int(sd))
        goal  = State(int(round(gx)), int(round(gy)), int(gd))
        if debug: print(f"[_load_edge_prims] computing detour A* start={start} goal={goal}")

        detour = self._astar_to_node(wm, belief, start, v, debug=debug)
        if detour is None:
            detour = []
        empty = (torch.is_tensor(detour) and detour.numel() == 0) or (not torch.is_tensor(detour) and len(detour) == 0)

        if empty:
            if debug: print("[_load_edge_prims] detour A* FAILED → empty")
            self.token_prims = []
        else:
            if debug:
                ln = detour.shape[0] if torch.is_tensor(detour) else len(detour)
                print(f"[_load_edge_prims] detour injected, len={ln}")
            self.token_prims = detour


    def step_plan(self, wm, belief, debug: bool = True):
        """
        Emit *one* primitive towards executing the current plan.

        Cases:
        1) Normal edge primitives (we are at u; cached link path known).
        2) Detour primitives (we injected an A* seq to v).
        3) If empty/unreachable, try forced detour; else mark plan_progress=-1.0.
        Returns: ([primitive_payload], 1) or ([], 0) on failure/done.
        """
        import torch

        # --- helpers that are SAFE for tensors or lists -----------------------
        def _plen(seq):
            if seq is None: return 0
            if torch.is_tensor(seq): return int(seq.shape[0])
            try: return len(seq)
            except Exception: return 0

        def _pempty(seq):
            if seq is None: return True
            if torch.is_tensor(seq): return seq.numel() == 0
            try: return len(seq) == 0
            except Exception: return True

        # ----------------------------------------------------------------------
        # 0) no plan or finished?
        # ----------------------------------------------------------------------
        if self.token_idx >= len(self.full_plan_tokens):
            if debug: print("[step_plan] plan complete or empty")
            self.plan_progress = 1.1 if self.full_plan_tokens else -1.0
            self.navigation_flags['plan_complete'] = True 
            return [], 0

        # ----------------------------------------------------------------------
        # 1) ensure we have primitives for *current* edge
        # ----------------------------------------------------------------------
        self._load_edge_primitives(wm, belief, debug=debug)

        # If still empty → decide whether to skip edge (maybe we are at v) or detour/fail
        while True:
            if not _pempty(self.token_prims):
                break  # good; we have something to emit

            # nothing loaded: are we already at this edge’s *target* node?
            u, v = self.full_plan_tokens[self.token_idx]
            if self._at_node_exact(v):
                if debug: print(f"[step_plan] already at target node {v} → advance edge")
                self.token_idx += 1
                if self.token_idx >= len(self.full_plan_tokens):
                    self.plan_progress = 1.1
                    self.navigation_flags['plan_complete'] = True 
                    return [], 0
                self._load_edge_primitives(wm, belief, debug=debug)
                continue

            # not at v & still empty → try forced detour to v
            if debug: print("[step_plan] empty but not at target → force detour inject")
            pose = self.get_current_pose()
            if pose is None:
                if debug: print("[step_plan] no pose; cannot detour → fail")
                self.plan_progress = -1.0
                return [], 0

            sx, sy, sd = pose
            gx, gy, gd = self.memory_graph.experience_map.get_pose(v)
            start = State(int(round(sx)), int(round(sy)), int(sd))
            goal  = State(int(round(gx)), int(round(gy)), int(gd))

            # guard: if start==goal (timing/rounding), let the while-loop recheck skip
            if (start.x, start.y, start.d) == (goal.x, goal.y, goal.d):
                if debug: print("[step_plan] start==goal; recheck skip on next loop")
                # leave token_prims empty and loop to the _at_node_exact(v) path
                continue

            detour = self.planner.astar_prims(wm, belief, start, goal, verbose=debug)
            if detour is None:
                detour = []

            if (torch.is_tensor(detour) and detour.numel() == 0) or \
            (not torch.is_tensor(detour) and len(detour) == 0):
                if debug: print("[step_plan] forced detour FAILED → abort plan")
                self.plan_progress = -1.0
                return [], 0

            if debug:
                ln = detour.shape[0] if torch.is_tensor(detour) else len(detour)
                print(f"[step_plan] forced detour injected len={ln}")
            self.token_prims = detour
            self.prim_idx = 0
            self._evade_injected = True
            break

        # ----------------------------------------------------------------------
        # 2) serve ONE primitive from token_prims
        # ----------------------------------------------------------------------
        raw_prim = self.token_prims[self.prim_idx]
        self.prim_idx += 1

        # normalize to string + payload
        if torch.is_tensor(raw_prim):
            idx = int(raw_prim.argmax().item())
            prim_str = ("forward", "right", "left")[idx]
            prim_out = raw_prim.cpu().tolist()
        elif isinstance(raw_prim, (list, tuple)):
            try:
                import numpy as _np
                idx = int(_np.argmax(raw_prim))
                prim_str = ("forward", "right", "left")[idx]
            except Exception:
                prim_str = str(raw_prim)
            prim_out = list(raw_prim)
        elif isinstance(raw_prim, str):
            prim_str = raw_prim
            prim_out = raw_prim
        else:
            prim_str = str(raw_prim)
            prim_out = raw_prim
        # normalize to string + payload
        if torch.is_tensor(raw_prim):
            idx = int(raw_prim.argmax().item())
            prim_str = ("forward", "right", "left")[idx]
            prim_out = raw_prim.cpu().tolist()
        elif isinstance(raw_prim, (list, tuple)):
            try:
                import numpy as _np
                idx = int(_np.argmax(raw_prim))
                prim_str = ("forward", "right", "left")[idx]
            except Exception:
                prim_str = str(raw_prim)
            prim_out = list(raw_prim)
        elif isinstance(raw_prim, str):
            prim_str = raw_prim
            prim_out = raw_prim
        else:
            prim_str = str(raw_prim)
            prim_out = raw_prim

        self._last_served_prim = prim_str

        # === ISSUE ACCOUNTING (so navigation_grade can see progress) ===
        self._issue_seq = int(getattr(self, "_issue_seq", 0)) + 1
        # capture a baseline pose and the pose_hist length at issue time
        hist = getattr(self, "_pose_hist", ())
        self._issue_baseline_hist_len = len(hist)
        if len(hist) > 0:
            self._issue_baseline_pose = hist[-1]
            self._issue_baseline_valid = True
        else:
            # fallback if pose_hist empty: use current pose if available
            cur = self.get_current_pose()
            if cur is not None:
                self._issue_baseline_pose = (float(cur[0]), float(cur[1]), float(cur[2]))
                self._issue_baseline_valid = True
            else:
                self._issue_baseline_valid = False
        self._last_served_prim = prim_str

        # ----------------------------------------------------------------------
        # 3) finished this primitive sequence?
        # ----------------------------------------------------------------------
        if self.prim_idx >= _plen(self.token_prims):
            if getattr(self, "_evade_injected", False):
                if debug: print("[step_plan] detour drained; will re-evaluate edge next tick")
                self.token_prims = []   # leave empty; skip/advance handled next call
                self.prim_idx = 0
                self._evade_injected = False
            else:
                # completed planned edge path
                self.token_idx += 1
                self.token_prims = []
                self.prim_idx = 0

        # ----------------------------------------------------------------------
        # 4) recompute progress scalar (0..1; don't mark -1 here)
        # ----------------------------------------------------------------------
        done_edges = self.token_idx
        fraction = (self.prim_idx / max(_plen(self.token_prims), 1)) if not _pempty(self.token_prims) else 0
        self.plan_progress = (done_edges + fraction) / max(len(self.full_plan_tokens), 1)

        if debug:
            print(f"[step_plan] edge={done_edges}/{len(self.full_plan_tokens)} "
                f"prim={self.prim_idx}/{_plen(self.token_prims)} "
                f"progress={self.plan_progress:.3f}")

        return [prim_out], 1




    def new_plan_from_grammar(self, grammar: PCFG):
        sent = random.choice(list(generate(grammar, depth=50)))
        self.full_plan_tokens = self._tokenize_plan(" ".join(sent))
        self.token_idx = self.prim_idx = 0
        self._reset_token_prims() 
        self.plan_progress = 0.0
        # final target = last rhs node
        if self.full_plan_tokens:
            _, self.target_node_id = self.full_plan_tokens[-1]

    def progress_scalar(self) -> float:
        """-1 no plan, 0-1 executing, 1.1 finished."""
        return self.plan_progress
    def task_solve_mission(self) -> str | None:
        """
        Return the mission string, e.g., "go to red room".
        First tries an attribute you can set externally (self.task_mission),
        then a generic fallback.
        """
        ms = getattr(self, "task_mission", None)
        if isinstance(ms, str) and ms.strip():
            return ms
        # Fallback – you can replace this if your env feeds a mission string elsewhere
        return "go to red room"
    def _parse_color_from_mission(self, mission: str | None) -> str | None:
        """
        Extract a MiniGrid color token from free text. Returns UPPERCASE or None.
        """
        if not mission:
            return None
        # canonical MiniGrid palette (+ a few aliases)
        canon = {
            "RED":"RED", "GREEN":"GREEN", "BLUE":"BLUE",
            "YELLOW":"YELLOW", "PURPLE":"PURPLE", "GREY":"GREY", "GRAY":"GREY",
            "BROWN":"BROWN", "ORANGE":"ORANGE"
        }
        m = re.search(r"\b(red|green|blue|yellow|purple|grey|gray|brown|orange)\b", mission, re.I)
        return canon.get(m.group(0).upper(), None) if m else None



    def push_pose(self, pose):
        """pose = (x,y,θ) or dict with x,y"""
        if pose is None:
            return
        if isinstance(pose, dict):
            self._pose_hist.append((float(pose["x"]), float(pose["y"]),float(pose["θ"])))
        else:
            self._pose_hist.append((float(pose[0]), float(pose[1]),float(pose[2])))
    def _at_node_exact(self, node_id: int) -> bool:
        pose = self.get_current_pose()
        if pose is None:
            return False
        x, y, th = pose
        nx, ny, nth = self.memory_graph.experience_map.get_pose(node_id)
        return int(round(x)) == int(nx) and int(round(y)) == int(ny) 
    def _coalesce(self, *vals):
        """Return the first value that is not None."""
        for v in vals:
            if v is not None:
                return v
        return None

    def _next_hop_after(self, v: int) -> int | None:
        """Return w if the next token is v->w, else None."""
        i = int(getattr(self, "token_idx", 0))
        if i + 1 < len(self.full_plan_tokens):
            u2, w = self.full_plan_tokens[i + 1]
            if u2 == v:
                return w
        return None

    def _heading_from_to(self, from_id: int, to_id: int | None) -> int | None:
        """
        Grid heading 0:E,1:N,2:W,3:S from pose deltas (x,y only).
        Safe when to_id is None or positions are identical (returns None).
        """
        if to_id is None:
            return None
        fx, fy, _ = self.memory_graph.experience_map.get_pose(from_id)
        tx, ty, _ = self.memory_graph.experience_map.get_pose(to_id)
        dx = int(round(tx - fx))
        dy = int(round(ty - fy))
        if dx == 0 and dy == 0:
            return None
        # Prefer axis with larger magnitude; ties fall back to x-axis
        if abs(dx) >= abs(dy):
            return 0 if dx > 0 else 2
        else:
            return 1 if dy > 0 else 3

    def _astar_to_node(self, wm, belief, start: "State", v: int, debug: bool = False):
        """
        Run A* from 'start' to node v, picking a *preferred* goal heading:
        • face from v toward next hop (if any),
        • else keep start.d (avoid gratuitous spins at final goal),
        • else fall back to stored gd at v.
        """
        gx, gy, gd_stored = self.memory_graph.experience_map.get_pose(v)

        # Compute preferred heading safely (don’t use boolean `or` with ints)
        preferred = self._heading_from_to(v, self._next_hop_after(v))
        goal_dir = self._coalesce(preferred, int(start.d) % 4, int(gd_stored) % 4)
        if goal_dir is None:
            goal_dir = int(gd_stored) % 4  # super-safe fallback

        goal = State(int(round(gx)), int(round(gy)), int(goal_dir))
        if debug:
            print(f"[_astar_to_node] v={v} start={start} goal={goal} (stored_gd={gd_stored}, next={self._next_hop_after(v)})")
        return self.planner.astar_prims(wm, belief, start, goal, verbose=debug)

    def universal_navigation(self, submode: str, wm, belief) -> tuple[list[str], int]:
        dbg = getattr(self, "debug_universal_navigation", False)
        def _log(msg: str) -> None:
            if dbg:
                print(msg)

        cur_pose = self.get_current_pose()
        _log(f"[universal_navigation] submode={submode} pose={cur_pose} "
            f"tok={self.token_idx}/{len(self.full_plan_tokens)} "
            f"prim_idx={self.prim_idx} prog={self.plan_progress:.3f}")

        # detour bookkeeping
        if not hasattr(self, "_evade_injected"):           self._evade_injected = False
        if not hasattr(self, "_evade_ticks_since_recalc"): self._evade_ticks_since_recalc = 0
        if not hasattr(self, "_evade_recalc_every"):
            import random; self._evade_recalc_every = random.randint(3, 5)

        # helpers
        def _is_empty_path(path) -> bool:
            if path is None: return True
            if hasattr(path, "numel"):
                try: return path.numel() == 0
                except Exception: pass
            try: return len(path) == 0
            except Exception: return False

        def _prims_pending() -> bool:
            seq = getattr(self, "token_prims", [])
            if seq is None: return False
            try:
                import torch
                if torch.is_tensor(seq):
                    return self.prim_idx < int(seq.shape[0])
            except Exception:
                pass
            try:
                return self.prim_idx < len(seq)
            except Exception:
                return False

        def _fallback_turn() -> tuple[list[str], int]:
            import random
            turn = random.choice(["right", "left"])
            self._last_served_prim = turn
            fb = [self.to_onehot_list(turn)]
            _log(f"[navigate] STALL → issuing {fb}")
            return fb, 1

        def _current_target():
            if self.token_idx < len(self.full_plan_tokens):
                return self.full_plan_tokens[self.token_idx]
            return None

        def _plan_local_path_to_target(tgt_exp_id):
            pose = self.get_current_pose()
            if pose is None:
                _log("[navigate] no current pose → cannot A*; returning EMPTY")
                return []
            start = self._pose_to_state(pose)
            tgt_pose = self.memory_graph.experience_map.get_pose(tgt_exp_id)
            goal  = self._pose_to_state(tgt_pose)
            _log(f"[navigate] A* request start={start} goal={goal} (tgt={tgt_exp_id})")
            try:
                path = self._astar_to_node(wm, belief, start, tgt_exp_id, debug=dbg)
                if _is_empty_path(path): _log("[navigate] A* result: EMPTY")
                else:
                    try: ln = len(path)
                    except Exception: ln = "<tensor>"
                    _log(f"[navigate] A* result: ok len={ln}")
                return path
            except Exception as e:
                _log(f"[navigate] astar_prims error: {e}")
                return []

        def _highlevel_replan_from_here(penalize_pair: tuple[int,int] | None = None,
                                        penalize_node: int | None = None) -> bool:
            if penalize_node is not None:
                _log("[navigate] note: node-confidence penalties are deprecated; ignoring.")
            try:
                if penalize_pair is not None:
                    u, v = int(penalize_pair[0]), int(penalize_pair[1])
                    for a, b in ((u, v), (v, u)):
                        link = self._find_link(a, b)
                        if link is not None:
                            link.confidence = 0
                    if not hasattr(self, "link_blacklist") or self.link_blacklist is None:
                        self.link_blacklist = set()
                    self.link_blacklist.add(frozenset({u, v}))
            except Exception as e:
                _log(f"[navigate] warn: link penalty failed: {e}")

            try:
                _log("[navigate] High-level REPLAN via PCFG…")
                grammar = self.build_pcfg_from_memory()
                self.generate_plan_with_pcfg(grammar)
                _log(f"[navigate] NEW full_plan_tokens={self.full_plan_tokens}")
                # Reset low-level progress
                self.token_prims = []
                self.prim_idx = 0
                self._evade_injected = False
                self._evade_ticks_since_recalc = 0
                ok = len(self.full_plan_tokens) > 0
                _log(f"[navigate] REPLAN success={ok}")
                return ok
            except Exception as e:
                _log(f"[navigate] High-level replan failed: {e}")
                return False

        # 0) If no plan at all, try one replan
        if not self.full_plan_tokens or self.token_idx >= len(self.full_plan_tokens):
            _log("[navigate] No usable plan tokens at entry.")
            if _highlevel_replan_from_here():
                _log("[navigate] Replan provided tokens; continuing.")
            else:
                _log("[navigate] Replan failed; falling back to stall-turn.")
                return _fallback_turn()

        # 1) Current edge + guarded reset
        cur_edge = _current_target()
        if cur_edge is None:
            _log("[navigate] current_target() is None after (re)plan.")
            return _fallback_turn()
        src, tgt = cur_edge
        if not hasattr(self, "_edge_key") or (src, tgt) != self._edge_key:
            self._edge_key = (src, tgt)
            self._edge_evades = 0
            self.navigation_flags.pop('evade_request', None)
            self.navigation_flags.pop('replan_request', None)
            self.navigation_flags.pop('replan_bad_node', None)
        _log(f"[navigate] current edge: src={src} -> tgt={tgt} key={getattr(self, '_edge_key', None)}")

        # 1a) ### FAST PATH: if we already have primitives to execute, just step them
        if _prims_pending():
            _log("[navigate] prims pending → step_plan()")
            prims, n = self.step_plan(wm, belief, debug=dbg)
            if not prims or n == 0:
                return _fallback_turn()
            return prims, n

        # 2) stalled motion escalates
        if self.navigation_flags.get('stalled_motion', False):
            _log("[navigate] stalled_motion → escalate to REPLAN")
            ok = _highlevel_replan_from_here(penalize_pair=(src, tgt), penalize_node=tgt)
            self.navigation_flags.pop('stalled_motion', None)
            if not ok:
                return _fallback_turn()
            cur_edge = _current_target()
            if cur_edge is None:
                return _fallback_turn()
            src, tgt = cur_edge
            if (src, tgt) != self._edge_key:
                self._edge_key = (src, tgt)
                self._edge_evades = 0
                self.navigation_flags.pop('evade_request', None)
                self.navigation_flags.pop('replan_request', None)
                self.navigation_flags.pop('replan_bad_node', None)

        # 3) explicit REPLAN request from grader
        if self.navigation_flags.get('replan_request', False):
            bad_node = self.navigation_flags.get('replan_bad_node', tgt)
            _log(f"[navigate] REPLAN requested; penalize node {bad_node} and pair ({src},{tgt})")
            ok = _highlevel_replan_from_here(penalize_pair=(src, tgt), penalize_node=bad_node)
            self.navigation_flags.pop('replan_request', None)
            self.navigation_flags.pop('replan_bad_node', None)
            self.navigation_flags.pop('evade_request', None)
            if not ok:
                return _fallback_turn()
            cur_edge = _current_target()
            if cur_edge is None:
                return _fallback_turn()
            src, tgt = cur_edge

        # 4) EVade request → inject/refresh a local detour
        if self.navigation_flags.get('evade_request', False) and not self._evade_injected:
            _log("[navigate] EVADE requested → compute local A* detour to tgt")
            if self._at_node_exact(tgt):
                _log(f"[navigate] already at target node {tgt} → skip edge via step_plan")
                prims, n = self.step_plan(wm, belief, debug=dbg)
                self.navigation_flags.pop('evade_request', None)
                if prims and n:
                    return prims, n
                cur_edge = _current_target()
                if cur_edge is None:
                    return _fallback_turn()
                src, tgt = cur_edge

            detour = _plan_local_path_to_target(tgt)
            if _is_empty_path(detour):
                _log("[navigate] EVADE A* empty → escalate to REPLAN")
                ok = _highlevel_replan_from_here(penalize_pair=(src, tgt), penalize_node=tgt)
                self.navigation_flags.pop('evade_request', None)
                if not ok:
                    return _fallback_turn()
                cur_edge = _current_target()
                if cur_edge is None:
                    return _fallback_turn()
                src, tgt = cur_edge
            else:
                try: ln = len(detour)
                except Exception: ln = "<tensor>"
                _log(f"[navigate] injecting detour len={ln}")
                self.token_prims = detour
                self.prim_idx = 0
                self._evade_injected = True
                self._evade_ticks_since_recalc = 0
                import random; self._evade_recalc_every = random.randint(3, 5)
                prims, n = self.step_plan(wm, belief, debug=dbg)
                if not prims or n == 0:
                    return _fallback_turn()
                self.navigation_flags.pop('evade_request', None)
                return prims, n

        # 5) Active detour: keep stepping / refreshing
        if self._evade_injected:
            self._evade_ticks_since_recalc += 1
            _log(f"[navigate] stepping detour; ticks={self._evade_ticks_since_recalc}/{self._evade_recalc_every}")

            if self._at_node_exact(tgt):
                _log(f"[navigate] reached target node {tgt} → resume main plan")
                self._evade_injected = False
                self.token_prims = []
                self.prim_idx = 0
                prims, n = self.step_plan(wm, belief, debug=dbg)
                if not prims or n == 0:
                    return _fallback_turn()
                return prims, n

            if self._evade_ticks_since_recalc >= self._evade_recalc_every:
                _log("[navigate] periodic A* refresh due")
                self._evade_ticks_since_recalc = 0
                import random; self._evade_recalc_every = random.randint(3, 5)
                detour = _plan_local_path_to_target(tgt)
                if not _is_empty_path(detour):
                    self.token_prims = detour
                    self.prim_idx = 0

            prims, n = self.step_plan(wm, belief, debug=dbg)
            if not prims or n == 0:
                return _fallback_turn()
            return prims, n

        # 6) No detour active → compute fresh local path (only if NO prims pending)
        _log("[navigate] no active detour → compute local path to tgt")
        if self._at_node_exact(tgt):
            _log(f"[navigate] already at target node {tgt} → skip edge via step_plan")
            prims, n = self.step_plan(wm, belief, debug=dbg)
            if prims and n:
                return prims, n
            cur_edge = _current_target()
            if cur_edge is None:
                return _fallback_turn()
            src, tgt = cur_edge

        detour = _plan_local_path_to_target(tgt)
        if _is_empty_path(detour):
            _log("[navigate] A* to current target FAILED → penalize and REPLAN")
            ok = _highlevel_replan_from_here(penalize_pair=(src, tgt), penalize_node=tgt)
            if not ok:
                return _fallback_turn()
            cur_edge = _current_target()
            if cur_edge is None:
                return _fallback_turn()
            src, tgt = cur_edge
            detour = _plan_local_path_to_target(tgt)
            if _is_empty_path(detour):
                _log("[navigate] Local path still empty after REPLAN → fallback turn")
                return _fallback_turn()

        try: ln = len(detour)
        except Exception: ln = "<tensor>"
        _log(f"[navigate] injecting local path len={ln}")
        self.token_prims = detour
        self.prim_idx = 0
        self._evade_injected = True
        self._evade_ticks_since_recalc = 0
        import random; self._evade_recalc_every = random.randint(3, 5)

        prims, n = self.step_plan(wm, belief, debug=dbg)
        if not prims or n == 0:
            return _fallback_turn()
        return prims, n




    # ---- small helper to safely format ints in logs (avoids NameError if you paste into class scope) ----
    def _safe_int(self,x):
        try:
            return int(x)
        except Exception:
            return x


    
    
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

    
    def _pose_to_state(self, pose):
        """Convert pose tuple to State object"""
        # Assuming State class exists as in your A* implementation
        x, y, direction = pose
        return State(int(round(x)), int(round(y)), int(direction))
    
    
    
    
# Observer integration (to be added to your step function)

    
    def navigation_grade(self) -> float:
        """
        Debuggable grader that is robust to prim_idx/token resets.
        Progress is keyed off a monotonic _issue_seq set by step_plan().
        """
        dbg = getattr(self, "debug_navigation_grade", True)
        def _log(*args):
            if dbg:
                print("[GRADE]", *args)

        mode = getattr(self, 'current_mode', 'NAVIGATE')
        if mode != 'NAVIGATE':
            if hasattr(self, "_edge_evades"): self._edge_evades = 0
            self.navigation_flags.pop('evade_request', None)
            self.navigation_flags.pop('replan_request', None)
            self.navigation_flags.pop('replan_bad_node', None)
            _log("mode != NAVIGATE → 0.0")
            return 0.0

        if self.token_idx >= len(self.full_plan_tokens):
            _log(f"no tokens to grade: token_idx={self.token_idx} len={len(self.full_plan_tokens)} → -1.0")
            return -1.0

        # NEW: sequence-based progress (ignore prim_idx churn)
        issue_seq = int(getattr(self, "_issue_seq", 0))
        prev_seq  = int(getattr(self, "_prev_issue_seq", -1))
        if issue_seq == prev_seq:
            _log(f"no new issued primitive yet (issue_seq={issue_seq}) → 0.0")
            return 0.0
        self._prev_issue_seq = issue_seq

        # Need at least two pose samples
        if len(getattr(self, "_pose_hist", ())) < 2:
            self._stall_bad_motion = 0
            _log("pose_hist < 2 → optimistic 1.0")
            return 1.0

        prim = getattr(self, "_last_served_prim", None)
        if prim is None:
            _log("WARN: _last_served_prim is None → 0.0")
            return 0.0

        # Choose baseline: prefer the one captured at issue time
        hist_len = len(self._pose_hist)
        x1, y1, th1 = self._pose_hist[-2]
        baseline = "hist[-2]"
        if getattr(self, "_issue_baseline_valid", False):
            bx, by, bth = self._issue_baseline_pose
            # use it only if at least one new sample was appended since issue
            if getattr(self, "_issue_baseline_hist_len", 0) <= hist_len - 1:
                x1, y1, th1 = bx, by, bth
                baseline = "issued_baseline"

        x2, y2, th2 = self._pose_hist[-1]

        x1i, y1i, th1i = int(round(x1)), int(round(y1)), int(round(th1))
        x2i, y2i, th2i = int(round(x2)), int(round(y2)), int(round(th2))

        if (x1i, y1i, th1i) == (x2i, y2i, th2i):
            _log(f"IDENTICAL samples {x1i,y1i,th1i} → graded two post-step poses. "
                f"Order must be: issue → execute → push_pose(new) → grade.")
            return 0.0

        # Current edge
        if self.token_idx < len(self.full_plan_tokens):
            u, v = self.full_plan_tokens[self.token_idx]
        else:
            u = v = None

        if not hasattr(self, "_edge_key"):
            self._edge_key = (u, v)
            self._edge_evades = 0
        if (u, v) != getattr(self, "_edge_key", None):
            _log(f"edge changed {(u,v)} (prev={getattr(self,'_edge_key', None)}) → reset per-edge flags")
            self._edge_key = (u, v)
            self._edge_evades = 0
            self.navigation_flags.pop('evade_request', None)
            self.navigation_flags.pop('replan_request', None)
            self.navigation_flags.pop('replan_bad_node', None)

        # Expected outcome
        def _expected_after(pose, action):
            xx, yy, dd = pose
            if action == "forward":
                dx, dy = DIR_VECS[dd % 4]
                return (xx + dx, yy + dy, dd)
            elif action == "left":
                return (xx, yy, (dd - 1) % 4)
            elif action == "right":
                return (xx, yy, (dd + 1) % 4)
            else:
                return (xx, yy, dd)

        ex_x, ex_y, ex_th = _expected_after((x1i, y1i, th1i), prim)
        ok_pos = (x2i == ex_x and y2i == ex_y)
        ok_dir = (th2i == ex_th)

        _log(f"prim='{prim}'  tok={self.token_idx}/{len(self.full_plan_tokens)}  "
            f"edge={(u,v)}  baseline={baseline}")
        _log(f"pose prev=({x1i},{y1i},{th1i})  curr=({x2i},{y2i},{th2i})  "
            f"expected=({ex_x},{ex_y},{ex_th})  ok_pos={ok_pos} ok_dir={ok_dir}")
        _log(f"flags pre: evades={getattr(self,'_edge_evades',0)}  "
            f"evade_req={self.navigation_flags.get('evade_request')}  "
            f"replan_req={self.navigation_flags.get('replan_request')}")

        if prim == "forward":
            if ok_pos and ok_dir:
                self.navigation_flags.pop('evade_request', None)
                self._edge_evades = 0
                self._stall_bad_motion = 0
                _log("FORWARD success → 1.0")
                return 1.0
            else:
                self.navigation_flags['evade_request'] = True
                self._edge_evades = int(getattr(self, "_edge_evades", 0)) + 1
                self._stall_bad_motion = int(getattr(self, "_stall_bad_motion", 0)) + 1
                thr = int(getattr(self, "evades_replan_threshold", 2))
                if self._edge_evades >= thr and v is not None:
                    self.navigation_flags['replan_request'] = True
                    self.navigation_flags['replan_bad_node'] = v
                    self.navigation_flags.pop('evade_request', None)
                    self.plan_progress = -1.0
                    _log(f"FORWARD fail (evades={self._edge_evades} >= {thr}) → replan_request, plan_progress=-1.0 → 0.0")
                else:
                    _log(f"FORWARD fail (evades={self._edge_evades}/{thr}) → evade_request → 0.0")
                return 0.0

        if prim in ("left", "right"):
            if ok_dir and (x2i == x1i) and (y2i == y1i):
                self._stall_bad_motion = 0
                _log(f"TURN {prim} success → 1.0")
                return 1.0
            else:
                self._stall_bad_motion = int(getattr(self, "_stall_bad_motion", 0)) + 1
                if self._stall_bad_motion >= int(getattr(self, "_stall_limit", 5)):
                    self.navigation_flags['stalled_motion'] = True
                    self.plan_progress = -1.0
                    _log(f"TURN {prim} fail (stall={self._stall_bad_motion}) → stalled_motion, plan_progress=-1.0 → 0.0")
                else:
                    _log(f"TURN {prim} minor fail (stall={self._stall_bad_motion}) → 0.0")
                return 0.0

        _log(f"unknown primitive '{prim}' → 0.5")
        return 0.5


    def _safe(self, u, v):
        try: return (int(u), int(v))
        except Exception: return (u, v)


    def _at_start_node_exact(self, start_id: int) -> bool:
        """True iff current pose *exactly* matches the stored node pose."""
        pose = self.get_current_pose()
        if pose is None:
            return False

        x, y, th = pose
        nx, ny, nth = self.memory_graph.experience_map.get_pose(start_id)

        # compare after rounding to the same precision that was used to store poses
        return int(round(x)) == int(nx) and \
            int(round(y)) == int(ny) 
    
    
          
    def get_primitives(self, u: int, v: int,wm,belief) -> list[str]:
        """
        Return the best primitive sequence for traversing the edge u→v:

        0) If we’ve already stored a path in the ExperienceLink, return it.
        1) Otherwise, build a fresh A* plan from *our real pose* → node v.
        2) Cache it in the link and return.
        """
        emap = self.memory_graph.experience_map

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
        prims = self._astar_to_node(wm, belief, start, v, debug=True)
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




    




# ──────────────────────────────────────────────────────────────────────────────
# Test harness: build surrogate cognitive graph from snapshot JSON and exercise
# build_pcfg_from_memory() + generate_plan_with_pcfg().
# Run:
#   python NavigationSystem.py --json /mnt/data/snapshot_t0480.json
# Options:
#   --infer-dist 3.4     (override inferred-link distance)
#   --seed 42            (deterministic link confidence synthesis)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, argparse, math, random, sys
    from collections import defaultdict

    # ----- Light-weight surrogate data structures to mimic your live objects ---
    class _SurLink:
        __slots__ = ("target", "confidence")
        def __init__(self, target, confidence=1):
            self.target = target
            self.confidence = int(confidence)

    class _SurExp:
        __slots__ = ("id", "x_m", "y_m", "dir", "links")
        def __init__(self, node):
            self.id   = int(node["id"])
            self.x_m  = float(node.get("x_m", node.get("x", 0.0)))
            self.y_m  = float(node.get("y_m", node.get("y", 0.0)))
            rp        = node.get("real_pose", None)
            self.dir  = int(rp[2]) if (isinstance(rp, (list, tuple)) and len(rp) >= 3) else int(node.get("dir", 0))
            self.links = []  # filled later

    class _SurExperienceMap:
        def __init__(self, exps):
            self.exps = exps
            self._pose = {e.id: (e.x_m, e.y_m, e.dir) for e in exps}
        def get_pose(self, exp_id: int):
            return self._pose[int(exp_id)]

    class _SurMemoryGraph:
        def __init__(self, exps, start_id):
            self.experience_map = _SurExperienceMap(exps)
            self._start_id = int(start_id)
        def get_current_exp_id(self) -> int:
            return self._start_id

    # ---------------------------- CLI -----------------------------------------
    ap = argparse.ArgumentParser(description="PCFG test harness over snapshot JSON")
    ap.add_argument("--json", default="/mnt/data/snapshot_t0480.json")
    ap.add_argument("--infer-dist", type=float, default=None,
                    help="Override inferred-link distance (if omitted, auto-derive)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ------------------------ Load snapshot -----------------------------------
    with open(args.json, "r") as f:
        snap = json.load(f)

    cog = snap.get("cog", {})
    nodes = cog.get("nodes", [])
    start_id = int(cog.get("current_exp_id", 0))

    if not nodes:
        print("[TEST] No nodes in snapshot; abort.", file=sys.stderr)
        sys.exit(2)

    # ----------------- Build surrogate experiences ----------------------------
    exps = [_SurExp(n) for n in nodes]
    id2exp = {e.id: e for e in exps}
    ids = sorted(id2exp.keys())

    # ---------- Synthesize a handful of explicit links with 0/1 confidence ----
    # Rule: connect each node → its next (by id) with conf=1, except every 10th
    # pair conf=0 (to verify permanent blacklisting). Also add the reverse link.
    dropped_pairs = []
    for i in range(len(ids) - 1):
        u, v = ids[i], ids[i+1]
        uexp, vexp = id2exp[u], id2exp[v]
        conf = 0 if ((u + v) % 10 == 0) else 1
        uexp.links.append(_SurLink(vexp, confidence=conf))
        vexp.links.append(_SurLink(uexp, confidence=conf))
        if conf == 0:
            dropped_pairs.append((u, v))

    # ------------------- Create memory graph ----------------------------------
    mg = _SurMemoryGraph(exps, start_id=start_id)

    # --------------- Current pose function (dock at start node) ---------------
    def _docked_pose():
        return mg.experience_map.get_pose(start_id)

    # --------------- Construct NavigationSystem instance ----------------------
    nav = NavigationSystem(memory_graph=mg, get_current_pose_func=_docked_pose, planner=None)

    # ------ Auto-derive a sensible inferred-link distance if not provided -----
    # Compute each node's nearest-neighbor distance; use 1.5× median as threshold.
    def _auto_infer_dist(exps_list):
        import statistics
        pts = [(e.x_m, e.y_m) for e in exps_list]
        dists = []
        for i, (xi, yi) in enumerate(pts):
            nd = float("inf")
            for j, (xj, yj) in enumerate(pts):
                if i == j: continue
                d = math.hypot(xi - xj, yi - yj)
                nd = min(nd, d)
            if math.isfinite(nd):
                dists.append(nd)
        if not dists:
            return 3.4
        med = statistics.median(dists)
        return 1.5 * med

    nav.infer_link_max_dist = float(args.infer_dist if args.infer_dist is not None else _auto_infer_dist(exps))

    # ----------------------------- RUN TEST -----------------------------------
    print("\n[TEST] ─────────────────────────────────────────────────────────────")
    print(f"[TEST] Nodes: {len(exps)}   Start ID: {start_id}   infer_link_max_dist: {nav.infer_link_max_dist:.3f}")
    if dropped_pairs:
        print(f"[TEST] Seeded {len(dropped_pairs)} zero-confidence links (will be blacklisted). Example: {dropped_pairs[:5]}")

    # Build grammar
    grammar = nav.build_pcfg_from_memory(debug=True)

    # Generate plan
    tokens = nav.generate_plan_with_pcfg(grammar, debug=True)

    # ----------------------------- SUMMARY ------------------------------------
    def _deg_stats(adj):
        degs = [len(v) for v in adj.values()]
        if not degs: return (0, 0, 0)
        return (min(degs), sum(degs)/len(degs), max(degs))

    adj = getattr(nav, "graph_used_for_pcfg", {})
    dmin, davg, dmax = _deg_stats(adj)

    print("\n[TEST] ===================== RESULT SUMMARY ========================")
    print(f"[TEST] Goal node id          : {getattr(nav, 'goal_node_id', None)}")
    print(f"[TEST] Plan tokens (u->v)    : {tokens}")
    print(f"[TEST] Token count           : {len(tokens)}")
    print(f"[TEST] Target node matches?  : {('YES' if (tokens and tokens[-1][1] == nav.goal_node_id) else 'NO/EMPTY')}")
    print(f"[TEST] Blacklist size        : {len(getattr(nav, 'link_blacklist', set()))}")
    print(f"[TEST] Adjacency |V|,|E|     : {len(adj)} nodes, {sum(len(v) for v in adj.values())} directed edges")
    print(f"[TEST] Degree stats (min/avg/max): {dmin:.0f}/{davg:.2f}/{dmax:.0f}")
    print("[TEST] =============================================================\n")
    # =========================== SECOND RUN ==================================
    # Scenario: treat every link as confidence==1, and DISABLE inferred links.
    # Blacklist remains honored (permanent) unless you explicitly override below.

    print("\n[TEST2] ────────────────────────────────────────────────────────────")
    print("[TEST2] Rebuilding PCFG with: all links treated as conf=1, no inferred links.")

    # Set toggles on the SAME nav instance (permanent blacklist remains in effect)
    nav.pcfg_treat_all_links_confident = True
    nav.pcfg_disable_inferred_links    = True
    # If you want to also ignore the blacklist in this pass, uncomment:
    # nav.pcfg_ignore_blacklist          = True

    grammar2 = nav.build_pcfg_from_memory(debug=True)
    tokens2  = nav.generate_plan_with_pcfg(grammar2, debug=True)

    adj2 = getattr(nav, "graph_used_for_pcfg", {})
    dmin2, davg2, dmax2 = _deg_stats(adj2)

    print("\n[TEST2] ===================== RESULT SUMMARY =======================")
    print(f"[TEST2] Goal node id          : {getattr(nav, 'goal_node_id', None)}")
    print(f"[TEST2] Plan tokens (u->v)    : {tokens2}")
    print(f"[TEST2] Token count           : {len(tokens2)}")
    print(f"[TEST2] Target node matches?  : {('YES' if (tokens2 and tokens2[-1][1] == nav.goal_node_id) else 'NO/EMPTY')}")
    print(f"[TEST2] Blacklist size        : {len(getattr(nav, 'link_blacklist', set()))}  (honored={not getattr(nav, 'pcfg_ignore_blacklist', False)})")
    print(f"[TEST2] Adjacency |V|,|E|     : {len(adj2)} nodes, {sum(len(v) for v in adj2.values())} directed edges")
    print(f"[TEST2] Degree stats (min/avg/max): {dmin2:.0f}/{davg2:.2f}/{dmax2:.0f}")
    print("[TEST2] =============================================================\n")
    print("\n[TEST3] ────────────────────────────────────────────────────────────")
    print("[TEST3] Rebuilding PCFG with: all links treated as conf=1, no inferred links.")

    # Set toggles on the SAME nav instance (permanent blacklist remains in effect)
    nav.pcfg_treat_all_links_confident = True
    nav.pcfg_disable_inferred_links    = False
    # If you want to also ignore the blacklist in this pass, uncomment:
    # nav.pcfg_ignore_blacklist          = True

    grammar3 = nav.build_pcfg_from_memory(debug=True)
    tokens3  = nav.generate_plan_with_pcfg(grammar2, debug=True)

    adj3 = getattr(nav, "graph_used_for_pcfg", {})
    dmin3, davg3, dmax3 = _deg_stats(adj3)

    print("\n[TEST2] ===================== RESULT SUMMARY =======================")
    print(f"[TEST2] Goal node id          : {getattr(nav, 'goal_node_id', None)}")
    print(f"[TEST2] Plan tokens (u->v)    : {tokens3}")
    print(f"[TEST2] Token count           : {len(tokens3)}")
    print(f"[TEST2] Target node matches?  : {('YES' if (tokens3 and tokens3[-1][1] == nav.goal_node_id) else 'NO/EMPTY')}")
    print(f"[TEST2] Blacklist size        : {len(getattr(nav, 'link_blacklist', set()))}  (honored={not getattr(nav, 'pcfg_ignore_blacklist', False)})")
    print(f"[TEST2] Adjacency |V|,|E|     : {len(adj3)} nodes, {sum(len(v) for v in adj3.values())} directed edges")
    print(f"[TEST2] Degree stats (min/avg/max): {dmin3:.0f}/{davg3:.2f}/{dmax3:.0f}")
    print("[TEST2] =============================================================\n")
