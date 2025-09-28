
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
def _vec(a, b):
    return (b[0]-a[0], b[1]-a[1])

def _norm(v):
    import math
    d = math.hypot(v[0], v[1])
    return (v[0]/d, v[1]/d) if d > 1e-9 else (0.0, 0.0)

def _dot(a,b): return a[0]*b[0] + a[1]*b[1]

def _edge_features(emap, a_id, b_id):
    ax, ay, _ = emap.get_pose(a_id)
    bx, by, _ = emap.get_pose(b_id)
    mid = ((ax+bx)*0.5, (ay+by)*0.5)
    dirv = _vec((ax,ay), (bx,by))
    return (ax, ay), (bx, by), mid, dirv

def _angdiff(u, v):
    import math
    nu, nv = _norm(u), _norm(v)
    dp = max(-1.0, min(1.0, _dot(nu, nv)))
    return math.degrees(math.acos(dp))

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight run exporter for hierarchical plans (JSON + FoV images)
# Paste near the top of control_eval/NavigationSystem.py (after imports).
# ──────────────────────────────────────────────────────────────────────────────
import os, json, time, math, datetime
from typing import Any

class PlanRunExporter:
    VALID_MODES = {"NAVIGATE", "TASK_SOLVING"}

    def __init__(self, root: str = "rans/plan_exports"):
        self.root = root
        self.run_dir: str | None = None
        self.events: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self._last_flush = 0.0
        self._step_counts: dict[tuple[int|None,int|None], int] = {}
        self._images_dir_name = "images"
        # NEW: optional hard override via env without touching any caller
        self._force_mode = os.environ.get("PLAN_EXPORT_FORCE_MODE", "").strip().upper() or None

        self._global_step = 0
        self.mode_session_idx = 1
        self.run_tag = "NAV1"  # default; will be set in .begin()

    # NEW: normalize any non-nav-ish mode to NAVIGATE
    def _normalize_mode(self, mode: str | None) -> str:
        m = str(mode or "NAVIGATE").upper()
        if self._force_mode in self.VALID_MODES:
            return self._force_mode
        return m if m in self.VALID_MODES else "NAVIGATE"

    # ---- public API ----------------------------------------------------------
    def begin(self, *, mode: str, mg, graph, inferred_pairs, tokens, goal_id, start_id, mode_session_idx: int | None = None) -> None:
        os.makedirs(self.root, exist_ok=True)
        mode = self._normalize_mode(mode)

        self.mode_session_idx = int(mode_session_idx or 1)
        short = "TS" if mode == "TASK_SOLVING" else "NAV"
        self.run_tag = f"{short}{self.mode_session_idx}"

        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.root, mode, stamp)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, self._images_dir_name), exist_ok=True)
        self._global_step = 0

        
        nodes = self._snapshot_nodes(mg)
        self.plan = {
            "version": "1.0",
            "created_utc": time.time(),
            "mode": mode,  # now guaranteed NAVIGATE or TASK_SOLVING
            "start_exp_id": int(start_id) if start_id is not None else None,
            "goal_node_id": int(goal_id) if goal_id is not None else None,
            "tokens": [list(map(int, t)) for t in (tokens or [])],
            "adjacency": {int(u): [int(v) for v in vs] for u,vs in (graph or {}).items()},
            "inferred_pairs": [list(map(int, p)) for p in (inferred_pairs or [])],
            "nodes": nodes,
            "nodes_media": self.plan.get("nodes_media", {}),
            "events": self.events,
        }
        # NEW: plan versioning (v0 = initial plan)
        self.plan["versions"] = [{
            "id": 0,
            "time": time.time(),
            "mode": mode,
            "run_tag": self.run_tag,
            "tokens": [list(map(int, t)) for t in (tokens or [])],
            "reason": {"kind": "initial"},
        }]
        self.plan["current_version"] = 0
        self._flush_json()

        # ---------- helpers ----------
    def _edge_key_tuple(self, edge):
        if not edge: return (None, None)
        try:
            u, v = edge
            return (int(u), int(v))
        except Exception:
            return (None, None)

    def _save_image_named(self, rel_subdir: str, filename: str, img) -> str | None:
        """
        Save under run_dir/<images>/<rel_subdir>/<filename>.png and return RELATIVE path.
        Accepts numpy HWC uint8, PIL.Image, or torch [3,H,W]/[H,W,3] in 0..1 or 0..255.
        """
        try:
            from PIL import Image
            import numpy as np, os
            sub = os.path.join(self.run_dir, self._images_dir_name, rel_subdir)
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, f"{filename}.png")

            # normalize to numpy uint8 HWC
            arr = img
            try:
                import torch
                if "torch" in str(type(arr)):
                    arr = arr.detach().cpu().numpy()
                    if arr.ndim == 3 and arr.shape[0] in (1,3):  # CHW -> HWC
                        arr = arr.transpose(1,2,0)
                    if arr.dtype != np.uint8:
                        arr = (arr * 255.0).clip(0,255).astype("uint8")
            except Exception:
                pass

            if "PIL" in str(type(arr)):
                arr.save(path)
            else:
                arr = np.asarray(arr)
                if arr.dtype != np.uint8:
                    arr = (arr * 255.0).clip(0,255).astype("uint8")
                Image.fromarray(arr).save(path)
            self._maybe_flush(force=True)
            return os.path.relpath(path, self.run_dir)
            
        except Exception:
            return None
    
    def bump_plan_version(self, *, prev_tokens: list[tuple[int,int]] | list[list[int]],
                        new_tokens: list[tuple[int,int]] | list[list[int]],
                        reason: dict | None = None) -> int:
        # normalize
        def _norm(tt):
            out = []
            for t in (tt or []):
                if isinstance(t, (list, tuple)) and len(t) == 2:
                    out.append((int(t[0]), int(t[1])))
            return out
        prev = _norm(prev_tokens)
        new  = _norm(new_tokens)

        # edge diffs for the visualizer
        prev_set = set(prev)
        new_set  = set(new)
        removed  = sorted(list(prev_set - new_set))
        added    = sorted(list(new_set  - prev_set))

        ver_id = int(self.plan.get("current_version", 0)) + 1
        entry = {
            "id": ver_id,
            "time": time.time(),
            "mode": self.plan.get("mode", "NAVIGATE"),
            "run_tag": self.run_tag,
            "tokens": [list(t) for t in new],
            "reason": reason or {"kind": "replan"},
            "diff": {"removed": [list(t) for t in removed], "added": [list(t) for t in added]},
        }
        self.plan.setdefault("versions", []).append(entry)
        self.plan["current_version"] = ver_id

        # convenient event for timelines
        self.log_event("replan_version", id=ver_id, reason=reason, added=entry["diff"]["added"], removed=entry["diff"]["removed"])
        self._maybe_flush(force=True)
        return ver_id


    # ---------- public: per-step FoV ----------
    def step_obs(self, *, edge: tuple[int,int] | None, token_idx: int,
                 pose: tuple[float,float,float] | None,
                 action: str | None,
                 image) -> None:
        """
        Save FoV for the current step grouped by edge, and log a 'step_obs' event.
        """
        # Optional throttle knobs (env or attribute)
        stride = int(getattr(self, "export_fov_stride", int(os.environ.get("PLAN_EXPORT_FOV_STRIDE", 1))))
        max_per_edge = int(getattr(self, "export_fov_max_per_edge", int(os.environ.get("PLAN_EXPORT_FOV_MAX_PER_EDGE", 999999))))

        ek = self._edge_key_tuple(edge)
        self._step_counts[ek] = self._step_counts.get(ek, 0) + 1
        seq = self._step_counts[ek]

        # throttle
        if (seq % max(1, stride)) != 0:
            return
        if seq > max_per_edge:
            return

        self._global_step += 1

        # folder name per-edge (or "phantom" when no edge)
        rel_subdir = "phantom" if ek == (None, None) else f"edge_{ek[0]}_{ek[1]}"
        # Example: TS1_step00012_tok0007_seq0003_forward
        fname = f"{self.run_tag}_step{self._global_step:05d}_tok{token_idx:04d}_seq{seq:04d}" + (f"_{action}" if action else "")
        rel = self._save_image_named(rel_subdir, fname, image) if image is not None else None
        # …log_event unchanged…
        info = {
            "token_idx": int(token_idx),
            "edge": list(ek) if ek != (None, None) else None,
            "pose": [float(p) for p in pose] if pose else None,
            "action": action,
            "image": rel,
        }
        self.log_event("step_obs", **info)

    # ---------- public: node media (snapshot + enhanced preds) ----------
    def node_created(self, *, node_id: int, snapshot_img, enhanced_pred_imgs: dict | list | tuple | None) -> dict:
        """
        Save creation-time snapshot and the 4 enhanced predictions for this node.
        Returns dict of relative paths and stores them under plan['nodes_media'][node_id].
        """
        if not self.run_dir:
            try:
                # Surface the mistake instead of silently returning None paths
                raise RuntimeError("PlanRunExporter.begin() was not called before node_created()")
            except Exception as e:
                # best-effort event
                try: self.log_event("node_media_attached_error", node_id=int(node_id), err=str(e))
                except Exception: pass
            return {"snapshot": None, "enhanced": {}}
        node_id = int(node_id)
        rels = {"snapshot": None, "enhanced": {}}

        # 1) snapshot
        if snapshot_img is not None:
            rel_snap = self._save_image_named(f"node_{node_id:05d}", "created_view", snapshot_img)           
            rels["snapshot"] = rel_snap
            # NEW: absolute
            rels["snapshot_abs"] = os.path.join(self.run_dir, rel_snap)
        
        # 2) enhanced (accept list[img] or dict[label->img])
        labels_default = ["L", "R", "LL", "RR"]
        if enhanced_pred_imgs:
            if isinstance(enhanced_pred_imgs, dict):
                items = enhanced_pred_imgs.items()
            else:
                items = zip(labels_default, list(enhanced_pred_imgs))
            for lab, im in items:
                rels["enhanced"][str(lab)] = self._save_image_named(f"node_{node_id:05d}", f"pred_{lab}", im)

        self.plan.setdefault("nodes_media", {})
        self.plan["nodes_media"][str(node_id)] = rels
        self._maybe_flush(force=True)
        self.log_event("node_media_attached", node_id=node_id, media=rels)
        return rels# ---------- public: node media (snapshot + enhanced preds) ----------
    
    def node_created(self, *, node_id: int, snapshot_img, enhanced_pred_imgs: dict | list | tuple | None) -> dict:
        """
        Save creation-time snapshot and the 4 enhanced predictions for this node.

        Keeps original behavior/keys:
        - returns/records relative paths under: {"snapshot": <rel>, "enhanced": {lab: <rel>, ...}}

        Adds (non-breaking):
        - "snapshot_abs":  absolute path to snapshot
        - "enhanced_abs":  {lab: absolute path} for each enhanced image
        """
        import os

        if not self.run_dir:
            try:
                raise RuntimeError("PlanRunExporter.begin() was not called before node_created()")
            except Exception as e:
                try:
                    self.log_event("node_media_attached_error", node_id=int(node_id), err=str(e))
                except Exception:
                    pass
            return {"snapshot": None, "enhanced": {}}

        node_id = int(node_id)
        rels: dict[str, Any] = {"snapshot": None, "enhanced": {}}

        run_root = os.path.abspath(self.run_dir)  # ensure absolute base

        # 1) snapshot (relative, plus absolute companion)
        if snapshot_img is not None:
            rel_snap = self._save_image_named(f"node_{node_id:05d}", "created_view", snapshot_img)
            rels["snapshot"] = rel_snap
            # NEW: absolute path (keeps relative key unchanged)
            rels["snapshot_abs"] = os.path.join(run_root, rel_snap) if rel_snap else None

        # 2) enhanced (accept list[img] or dict[label->img])
        labels_default = ["L", "R", "LL", "RR"]
        if enhanced_pred_imgs:
            if isinstance(enhanced_pred_imgs, dict):
                items = enhanced_pred_imgs.items()
            else:
                items = zip(labels_default, list(enhanced_pred_imgs))
            for lab, im in items:
                rel_p = self._save_image_named(f"node_{node_id:05d}", f"pred_{lab}", im)
                rels["enhanced"][str(lab)] = rel_p

        # NEW: absolute companions for enhanced (non-breaking addition)
        rels["enhanced_abs"] = {
            str(lab): os.path.join(run_root, rel_p)
            for lab, rel_p in rels["enhanced"].items()
            if rel_p
        }

        # Record into plan.json exactly as before (now with extra *_abs keys present)
        self.plan.setdefault("nodes_media", {})
        self.plan["nodes_media"][str(node_id)] = rels

        self._maybe_flush(force=True)
        self.log_event("node_media_attached", node_id=node_id, media=rels)
        return rels



    def log_event(self, kind: str, **kw) -> None:
        evt = {"t": time.time(), "type": kind}
        evt.update(kw)
        self.events.append(evt)
        self._maybe_flush()

    def node_reached(self, node_id: int, pose: tuple[float,float,float] | None,
                     get_fov_image_fn=None) -> None:
        info = {"node_id": int(node_id)}
        if pose:
            x,y,th = pose
            info["pose"] = [float(x), float(y), float(th)]
        # Save FoV if we can
        img_rel = None
        if callable(get_fov_image_fn):
            try:
                img = get_fov_image_fn()
                img_rel = self._save_image(node_id, img)
            except Exception as e:
                info["fov_error"] = f"{e}"
        if img_rel:
            info["fov_image"] = img_rel
        self.log_event("node_reached", **info)

    # ---- internals -----------------------------------------------------------
    def _snapshot_nodes(self, mg) -> dict[str, dict]:
        try:
            emap = mg.experience_map
            out: dict[str, dict] = {}
            for e in getattr(emap, "exps", []):
                nid = int(e.id)
                x = getattr(e, "x", None); y = getattr(e, "y", None)
                if x is None or y is None:
                    try:
                        xx, yy, dd = emap.get_pose(nid)
                        x, y = float(xx), float(yy)
                        pose = [float(xx), float(yy), float(dd)]
                    except Exception:
                        pose = None
                else:
                    pose = None
                node = {
                    "id": nid,
                    "x": float(x) if x is not None else None,
                    "y": float(y) if y is not None else None,
                }
                if pose is not None:
                    node["pose"] = pose
                for k in ("place_kind", "room_color", "confidence"):
                    if hasattr(e, k):
                        node[k] = getattr(e, k)

                # NEW: surface media paths if present on Experience
                if hasattr(e, "media_snapshot_path"):
                    node["snapshot_abs"] = str(getattr(e, "media_snapshot_path"))
                if hasattr(e, "media_enhanced_preds_paths"):
                    node["enhanced_preds_abs"] = list(map(str, getattr(e, "media_enhanced_preds_paths")))

                out[str(nid)] = node
            return out
        except Exception:
            return {}


    def _save_image(self, node_id: int, img) -> str | None:
        # Accept numpy (H,W,3) uint8, PIL.Image, or torch.Tensor [H,W,3] / [3,H,W]
        path = os.path.join(self.run_dir, "images", f"node_{int(node_id):05d}.png")
        try:
            from PIL import Image
            import numpy as np
            try:
                import torch
                if "torch" in str(type(img)):
                    arr = img.detach().cpu().numpy()
                    if arr.ndim == 3 and arr.shape[0] in (1,3):
                        arr = arr.transpose(1,2,0)
                    arr = (arr * 255.0).clip(0,255).astype('uint8') if arr.dtype != np.uint8 else arr
                    Image.fromarray(arr).save(path)
                    rel = os.path.relpath(path, self.run_dir)
                    self._maybe_flush(force=True)
                    return rel
            except Exception:
                pass
            if isinstance(img, Image.Image):
                img.save(path)
            else:
                arr = np.asarray(img)
                if arr.dtype != np.uint8:
                    arr = (arr * 255.0).clip(0,255).astype('uint8')
                Image.fromarray(arr).save(path)
            rel = os.path.relpath(path, self.run_dir)
            self._maybe_flush(force=True)
            return rel
        except Exception:
            return None

    def _flush_json(self) -> None:
        if not self.run_dir:
            return
        try:
            self.plan["events"] = self.events
            with open(os.path.join(self.run_dir, "plan.json"), "w", encoding="utf-8") as f:
                json.dump(self.plan, f, indent=2)
            self._last_flush = time.time()
        except Exception:
            pass
    

    def _maybe_flush(self, force: bool = False) -> None:
        if force or (time.time() - self._last_flush) > 1.0:
            self._flush_json()




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
        self._mode_entry_counts = {"NAVIGATE": 0, "TASK_SOLVING": 0}
        self._mode_session_idx = 0


        self.plan_export = PlanRunExporter(root=getattr(self, "plan_export_root", "runs/plan_exports"))
        self._export_last_token_idx = -1  # for safe node-arrival detection


        self.task_runs_completed = 0        # count of completed TASK_SOLVING runs
        self.task_mutation_armed = False    # one-shot latch for the outer loop

        # --- post-finish hold (ticks we will not replan) ---
        self.plan_finished_hold_ticks = int(getattr(self, "plan_finished_hold_ticks", 2))
        self._post_finish_hold = 0

        self._need_replan_on_enter = False

        self._visit_counts = defaultdict(int)   # id -> visits
        self.goal_visit_hard_limit = getattr(self, "goal_visit_hard_limit", 2)
        # --- session & partial-path bookkeeping ---
        self._last_mode = None                 # to detect re-entry into NAVIGATE
        self._nav_session_id = 0
        self._partial_attempts = defaultdict(int)  # key: (u,v) → retries
        self.partial_retries_before_replan = 6     # configurable
        self.max_partial_len = 8                   # configurable small partial detours
        self._last_edge = None
        # ─── NEW: “phantom” final-push state ───
        self._phantom_goal_pose: tuple[int,int,int] | None = None  # (x,y,θ) in grid coords
        self._phantom_active: bool = False

        # knobs (tweakable, safe defaults)
        self.phantom_cell_grid = 4                # NxN zoning grid
        self.phantom_max_goal_cell_cheby = 1      # only consider empty/sparse cells Chebyshev-adjacent to GOAL’s cell
        self.phantom_min_gap_dist = 6.0           # reject phantom if centroid is too close to known nodes
        self.phantom_partial_len = 6              # cap pushed partial A* length
        self._phantom_attempts = 0
        self.phantom_retries_before_finish = getattr(self, "phantom_retries_before_finish", 6)


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
    def _ensure_portal_store(self):
        if not hasattr(self, "_closed_portals") or self._closed_portals is None:
            # Each item: dict(mid=(x,y), dir=(dx,dy), half_len=float, half_w=float)
            self._closed_portals = []

    def _register_portal_closure(self, u, v, dbg=print):
        """
        Called when edge (u,v) proved impassable. Uses node poses only.
        Blacklists *all* edges whose midpoints & directions match the same portal strip.
        Also records a portal descriptor so inferred links get filtered later.
        """
        mg = self.memory_graph
        emap = mg.experience_map

        # Geometry of failed edge
        (ax,ay), (bx,by), mid_uv, dir_uv = _edge_features(emap, u, v)
        import math
        L = math.hypot(dir_uv[0], dir_uv[1]) + 1e-9
        dir_uv = (dir_uv[0]/L, dir_uv[1]/L)            # unit direction
        # Tunables (no map metadata required)
        ang_tol_deg   = float(getattr(self, "portal_angle_tol_deg", 25.0))
        half_len_pad  = float(getattr(self, "portal_half_len_pad", 1.0))   # how far along the doorway
        half_w        = float(getattr(self, "portal_half_width", 1.2))     # doorway lateral half-width
        half_len      = (0.5*L) + half_len_pad

        # Persist the portal descriptor for future filtering (inferred links)
        self._ensure_portal_store()
        self._closed_portals.append({
            "mid": mid_uv, "dir": dir_uv, "half_len": half_len, "half_w": half_w
        })

        # Helper: does edge (a,b) lie in the same portal strip?
        def edge_hits_portal(a,b):
            (pax,pay),(pbx,pby), mid_ab, dir_ab = _edge_features(emap, a, b)
            # angular match
            if _angdiff(dir_uv, dir_ab) > ang_tol_deg:
                return False
            # project midpoint difference onto portal frame
            dx, dy = (mid_ab[0]-mid_uv[0], mid_ab[1]-mid_uv[1])
            # longitudinal along dir_uv, lateral across it
            # perp = rotate90(dir_uv)
            perp = (-dir_uv[1], dir_uv[0])
            lon = abs(dx*dir_uv[0] + dy*dir_uv[1])
            lat = abs(dx*perp[0]   + dy*perp[1])
            return (lon <= half_len) and (lat <= half_w)

        # Collect all existing edges (both directions) and sever the ones that hit the same portal
        if not hasattr(self, "link_blacklist") or self.link_blacklist is None:
            self.link_blacklist = set()
        canon = lambda a,b: frozenset((a,b))
        severed = 0

        # Iterate all known links in the memory graph
        exps = getattr(emap, "exps", [])
        for e in exps:
            for lnk in getattr(e, "links", []):
                tgt = getattr(lnk, "target", None)
                vid = getattr(tgt, "id", None) if tgt is not None else getattr(lnk, "target_id", None)
                if vid is None: 
                    continue
                if edge_hits_portal(e.id, vid):
                    # drop confidence on both directions if present
                    l = self._find_link(e.id, vid)
                    if l is not None: setattr(l, "confidence", 0)
                    l = self._find_link(vid, e.id)
                    if l is not None: setattr(l, "confidence", 0)
                    # and permanently blacklist the pair
                    self.link_blacklist.add(canon(e.id, vid))
                    severed += 1

        if getattr(self, "debug_universal_navigation", False):
            dbg(f"[portal] closed around ({u},{v}) → severed {severed} links; "
                f"portal mid={mid_uv} half_len={half_len:.2f} half_w={half_w:.2f} ang_tol={ang_tol_deg}°")


    def _heading_from_vector(self, dx: float, dy: float) -> int:
        """Grid heading 0:E,1:N,2:W,3:S from (dx,dy)."""
        if abs(dx) >= abs(dy):
            return 0 if dx > 0 else 2
        else:
            return 1 if dy > 0 else 3
    def _predict_phantom_pose_using_zones(self, exps, start_id: int, goal_id: int,
                                      graph: dict[int, list[int]] | None = None,
                                      debug: bool = False) -> tuple[int,int,int] | None:
        """
        Phantom targeter (goal-agnostic) aligned with _pcfg_select_goal_node_scored:

        1) Build the same 4×4 grid and compute interest per cell:
        interest = 0.5*deficit_vs_median + 0.35*scarcity + 0.15*frontier
        2) Pick the single best cell by interest (optionally tie-broken by outwardness).
        3) Infer a global lattice (step + phase) from node deltas and place candidates
        at lattice intersections *inside that cell*; pick the one with the largest min-gap
        to existing nodes (>= phantom_min_gap_dist).
        4) If lattice fails, fallback to cell centroid (min-gap guarded).
        5) Heading = direction (E/N/W/S) toward the emptiest neighboring cell around the chosen cell.
        """
        import math
        from collections import Counter

        # --- basic grid helpers ---
        Nx, Ny, bbox, step_x, step_y, cell_of, centroid, counts, _ = self._zone_grid(exps)
        min_x, max_x, min_y, max_y = bbox

        def _xy(e):
            return float(getattr(e, "x_m", getattr(e, "x", 0.0))), float(getattr(e, "y_m", getattr(e, "y", 0.0)))

        nodes = [(int(e.id), *_xy(e)) for e in exps]
        if len(nodes) < 4 or step_x <= 0 or step_y <= 0:
            if debug:
                print("[PHANTOM] insufficient nodes or degenerate steps → None")
            return None

        id2xy = {nid: (x, y) for nid, x, y in nodes}

        # --- interest terms (same blend as _pcfg_select_goal_node_scored) ---
        flat = [counts[i][j] for i in range(Nx) for j in range(Ny)]
        nonzero = [c for c in flat if c > 0]
        # scarcity
        scarcity_cell = [[1.0 / (1.0 + counts[i][j]) for j in range(Ny)] for i in range(Nx)]
        # deficit vs median(nonzero)
        med_nz = (sorted(nonzero)[len(nonzero)//2] if nonzero else 0)
        med_nz = max(med_nz, 1)
        deficit_cell = [[max(0.0, (med_nz - counts[i][j]) / med_nz) for j in range(Ny)] for i in range(Nx)]
        # frontier = fraction of EMPTY neighbors in 3×3 ring
        empty = {(i, j) for i in range(Nx) for j in range(Ny) if counts[i][j] == 0}
        def frontier_frac(ix, iy):
            tot = 0; emp = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    x2, y2 = ix + dx, iy + dy
                    if 0 <= x2 < Nx and 0 <= y2 < Ny:
                        tot += 1
                        if (x2, y2) in empty: emp += 1
            return (emp / tot) if tot > 0 else 0.0
        frontier_cell = [[frontier_frac(i, j) for j in range(Ny)] for i in range(Nx)]

        a_def, a_scar, a_front = 0.5, 0.35, 0.15
        interest_cell = [[
            a_def   * deficit_cell[i][j] +
            a_scar  * scarcity_cell[i][j] +
            a_front * frontier_cell[i][j]
        for j in range(Ny)] for i in range(Nx)]

        # --- choose best cell by interest (tie-break by outwardness) ---
        def outward(ix, iy):
            edge = min(ix, Nx-1-ix, iy, Ny-1-iy)
            rng  = max(Nx, Ny)
            return 1.0 - (edge / max(rng/2.0, 1.0))

        all_cells = [(ix, iy) for ix in range(Nx) for iy in range(Ny)]
        # Prefer empty/sparse implicitly via interest; tie-break with outward
        all_cells.sort(key=lambda ij: (interest_cell[ij[0]][ij[1]], outward(ij[0], ij[1])), reverse=True)
        best_ix, best_iy = all_cells[0]
        cx, cy = centroid(best_ix, best_iy)

        if debug:
            print(f"[PHANTOM] grid={Nx}x{Ny} step=({step_x:.2f},{step_y:.2f}) bbox=({min_x:.2f},{min_y:.2f})→({max_x:.2f},{max_y:.2f})")
            print(f"[PHANTOM] chosen cell=({best_ix},{best_iy}) interest={interest_cell[best_ix][best_iy]:.3f} "
                f"deficit={deficit_cell[best_ix][best_iy]:.3f} scarcity={scarcity_cell[best_ix][best_iy]:.3f} "
                f"frontier={frontier_cell[best_ix][best_iy]:.3f} outward={outward(best_ix,best_iy):.3f}")

        # --- infer lattice (step + phase) from global node deltas ---
        def infer_steps_and_phase():
            xs = []; ys = []
            def push_pair(u, v):
                x1, y1 = id2xy[u]; x2, y2 = id2xy[v]
                dx = int(round(x2 - x1)); dy = int(round(y2 - y1))
                if dx: xs.append(abs(dx))
                if dy: ys.append(abs(dy))

            if graph:
                for u, nbs in graph.items():
                    for v in nbs:
                        push_pair(u, v)
            if not xs and not ys:
                # kNN deltas (3 nearest)
                pts = list(id2xy.items())
                for i, (ui, (x1, y1)) in enumerate(pts):
                    neigh = sorted(((math.hypot(x1 - x2, y1 - y2), uj)
                                    for uj, (x2, y2) in pts if uj != ui))[:3]
                    for _, uj in neigh:
                        push_pair(ui, uj)

            def mode_or_default(arr, default_val):
                if not arr: return default_val
                c = Counter(arr)
                v, _ = c.most_common(1)[0]
                return int(max(1, min(6, v)))  # clamp to [1,6] cells by default

            dx_mode = mode_or_default(xs, default_val=max(1, int(round(step_x))))
            dy_mode = mode_or_default(ys, default_val=max(1, int(round(step_y))))

            # Phase via histogram on (coord mod step)
            def phase(values, step):
                if step <= 0: return 0
                bins = Counter(int(round(v % step)) for v in values)
                return max(bins.items(), key=lambda kv: kv[1])[0] if bins else 0

            xs_all = [x for _, x, _ in nodes]
            ys_all = [y for _, _, y in nodes]
            px = phase(xs_all, dx_mode)
            py = phase(ys_all, dy_mode)
            return dx_mode, dy_mode, px, py

        dx_mode, dy_mode, phase_x, phase_y = infer_steps_and_phase()
        if debug:
            print(f"[PHANTOM] inferred lattice: step=({dx_mode},{dy_mode}) phase=({phase_x},{phase_y})")

        # --- generate lattice intersections inside the chosen cell ---
        # cell bounds:
        x0 = min_x + best_ix * step_x
        x1 = x0 + step_x
        y0 = min_y + best_iy * step_y
        y1 = y0 + step_y

        def lattice_points_in_cell():
            pts = []
            if dx_mode > 0 and dy_mode > 0:
                # search k,l such that x=k*dx+phase_x in [x0,x1], y=l*dy+phase_y in [y0,y1]
                k_min = int(math.floor((x0 - phase_x) / dx_mode)) - 1
                k_max = int(math.ceil ( (x1 - phase_x) / dx_mode)) + 1
                l_min = int(math.floor((y0 - phase_y) / dy_mode)) - 1
                l_max = int(math.ceil ( (y1 - phase_y) / dy_mode)) + 1
                for k in range(k_min, k_max + 1):
                    lx = k * dx_mode + phase_x
                    if lx < x0 - 1e-6 or lx > x1 + 1e-6: continue
                    for l in range(l_min, l_max + 1):
                        ly = l * dy_mode + phase_y
                        if ly < y0 - 1e-6 or ly > y1 + 1e-6: continue
                        pts.append((int(round(lx)), int(round(ly))))
            return pts

        cand_pts = lattice_points_in_cell()

        # Always consider the exact cell centroid as a fallback candidate
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if (cx_i, cy_i) not in cand_pts:
            cand_pts.append((cx_i, cy_i))

        # --- score candidates by gap-to-known (max-min-distance) + centeredness ---
        # --- score candidates by gap-to-known (max-min-distance) + centeredness ---
        # Dynamic min_gap: cap by a fraction of the cell size so dense maps still allow a phantom
        base_gap   = float(getattr(self, "phantom_min_gap_dist", 5.0))
        gap_frac   = float(getattr(self, "phantom_gap_as_fraction_of_cell", 0.60))  # new knob
        cell_scale = max(step_x, step_y, 1.0)
        min_gap    = min(base_gap, gap_frac * cell_scale)

        def gap_to_known(px, py):
            if not nodes: return float('inf')
            return min(math.hypot(px - x, py - y) for _, x, y in nodes)
        # normalize helpers
        norm_gap = max(step_x, step_y, 1.0)
        def score_point(px, py):
            g = gap_to_known(px, py)
            center_bias = 1.0 / (1.0 + math.hypot(px - cx, py - cy))
            # prioritize min-gap first, then center bias
            return (g / norm_gap, center_bias)

        scored_pts = []
        for (px, py) in cand_pts:
            g = gap_to_known(px, py)
            if g >= min_gap:
                sc = score_point(px, py)
                scored_pts.append((sc, px, py))

        if not scored_pts:
            if debug:
                print(f"[PHANTOM] no lattice candidate meets min_gap={min_gap:.2f}; trying centroid/relax…")
            g = gap_to_known(cx_i, cy_i)

            # Relaxed acceptance
            relax = float(getattr(self, "phantom_gap_relax_factor", 0.60))
            if g >= relax * min_gap:
                chosen = (cx_i, cy_i)
            else:
                # Nudge the centroid toward the emptiest neighbor cell
                nudge = float(getattr(self, "phantom_nudge_frac", 0.33)) * cell_scale
                # best_dir already points to emptiest neighbor
                dx = [ nudge, 0.0, -nudge, 0.0 ][best_dir]   # E,N,W,S
                dy = [ 0.0,  nudge, 0.0, -nudge][best_dir]
                nx, ny = int(round(cx + dx)), int(round(cy + dy))
                chosen = (nx, ny)

        else:
            scored_pts.sort(reverse=True)  # max gap, then center bias
            _, px, py = scored_pts[0]
            chosen = (px, py)
        if debug and scored_pts:
            top = scored_pts[:3]
            print(f"[PHANTOM] top candidates by (gap/norm, center_bias): {[(p[1], p[2], p[0][0], round(p[0][1],3)) for p in top]}")

        # --- heading: toward emptiest neighbor (E=0,N=1,W=2,S=3)
        def neighbor_emptiness(ix, iy):
            def empt(i, j):
                if 0 <= i < Nx and 0 <= j < Ny:
                    # higher = emptier
                    return 1.0 / (1.0 + counts[i][j])
                return 0.0
            return {
                0: empt(ix + 1, iy),  # E
                1: empt(ix, iy + 1),  # N
                2: empt(ix - 1, iy),  # W
                3: empt(ix, iy - 1),  # S
            }

        neigh = neighbor_emptiness(best_ix, best_iy)
        best_dir = max(neigh.items(), key=lambda kv: kv[1])[0]
        # If chosen cell is crowded, and there is an empty neighbor cell, bias heading toward that empty cell
        if counts[best_ix][best_iy] > 0:
            neigh = {  # E,N,W,S
                (best_ix+1, best_iy),
                (best_ix, best_iy+1),
                (best_ix-1, best_iy),
                (best_ix, best_iy-1),
            }
            empty_neighbors = [(i,j) for (i,j) in neigh if 0 <= i < Nx and 0 <= j < Ny and counts[i][j] == 0]
            if empty_neighbors:
                # nudge heading toward emptiest neighbor (already computed as best_dir)
                pass  # heading already points to emptiest; this keeps behavior deterministic
        if debug:
            print(f"[PHANTOM] candidates={len(cand_pts)} kept={len(scored_pts)} "
                f"min_gap={min_gap:.2f} chosen=({chosen[0]},{chosen[1]}) "
                f"heading={best_dir} (E=0,N=1,W=2,S=3) "
                f"neighbor_emptiness={neigh}")

        return (int(chosen[0]), int(chosen[1]), int(best_dir))

    
    def _zone_grid(self, exps, Nx: int | None = None, Ny: int | None = None):
        """Build a simple Nx×Ny grid over node bbox and return helpers."""
        import math
        Nx = Nx or int(getattr(self, "phantom_cell_grid", 4))
        Ny = Ny or Nx
        xs = [float(getattr(e, "x_m", getattr(e, "x", 0.0))) for e in exps]
        ys = [float(getattr(e, "y_m", getattr(e, "y", 0.0))) for e in exps]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        step_x = (max_x - min_x) / max(Nx, 1) or 1.0
        step_y = (max_y - min_y) / max(Ny, 1) or 1.0

        def cell_of(x, y):
            ix = min(Nx-1, max(0, int((x - min_x) / step_x))) if step_x > 0 else 0
            iy = min(Ny-1, max(0, int((y - min_y) / step_y))) if step_y > 0 else 0
            return ix, iy

        def centroid(ix, iy):
            return (min_x + (ix + 0.5) * step_x, min_y + (iy + 0.5) * step_y)

        counts = [[0 for _ in range(Ny)] for _ in range(Nx)]
        by_cell = {}
        for e in exps:
            x = float(getattr(e, "x_m", getattr(e, "x", 0.0)))
            y = float(getattr(e, "y_m", getattr(e, "y", 0.0)))
            ij = cell_of(x, y)
            counts[ij[0]][ij[1]] += 1
            by_cell.setdefault(ij, []).append(e.id)

        return Nx, Ny, (min_x, max_x, min_y, max_y), step_x, step_y, cell_of, centroid, counts, by_cell

    def _astar_to_pose_partial(self, wm, belief, start: "State",
                            pose_xyz: tuple[int,int,int],
                            debug: bool = False,
                            max_partial_len: int | None = None):
        """Partial A* to an arbitrary grid pose (x,y,θ) — used for phantom pushes."""
        if max_partial_len is None:
            max_partial_len = int(getattr(self, "phantom_partial_len", 6))
        gx, gy, gd = pose_xyz
        goal = State(int(round(gx)), int(round(gy)), int(gd))
        try:
            ret = self.planner.astar_prims(wm, belief, start, goal, verbose=debug, allow_partial=True)
            if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
                path, meta = ret
            else:
                path, meta = ret, {'reached_goal': True, 'best_dist': None}
        except TypeError:
            # planner without partial support
            path = self.planner.astar_prims(wm, belief, start, goal, verbose=debug)
            meta = {'reached_goal': True, 'best_dist': None}

        # normalize and cap
        import torch
        if path is None or (torch.is_tensor(path) and path.numel() == 0) or (hasattr(path, "__len__") and len(path) == 0):
            return [], {'reached_goal': False, 'best_dist': None}
        try:
            if hasattr(path, "shape") and path.shape[0] > max_partial_len:
                path = path[:max_partial_len]
            elif hasattr(path, "__len__") and len(path) > max_partial_len:
                path = path[:max_partial_len]
        except Exception:
            pass
        return path, meta


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
    

    # ---------------------------------------------------------------------
    # Mission helpers
    # ---------------------------------------------------------------------
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
    def emit_stall_turn(self, *, for_executor: bool = False, preferred: str | None = None):
        """
        Emit a single left/right turn.

        - If for_executor=True: DO NOT bump _issue_seq → navigation_grade() will return 0.0.
        Use this when you must output something to avoid crashing an outer loop.
        - If for_executor=False: behaves like the internal stall (counts for grade).

        Returns: ( [onehot], 1, label_str )
        """
        import random
        turn = preferred if preferred in ("left", "right") else random.choice(["left", "right"])
        self._last_served_prim = turn

        # Only count for grading if we are *intentionally* stalling inside navigation.
        if not for_executor:
            self._issue_seq = int(getattr(self, "_issue_seq", 0)) + 1

        # Baseline bookkeeping (same as the inner helper)
        hist = getattr(self, "_pose_hist", ())
        self._issue_baseline_hist_len = len(hist)
        if len(hist) > 0:
            self._issue_baseline_pose = hist[-1]
            self._issue_baseline_valid = True
        else:
            cur = self.get_current_pose()
            if cur is not None:
                self._issue_baseline_pose = (float(cur[0]), float(cur[1]), float(cur[2]))
                self._issue_baseline_valid = True
            else:
                self._issue_baseline_valid = False

        # Optional: tag it; not required, but useful for debugging
        if for_executor:
            self.navigation_flags['synthetic_action'] = True

        fb = [self.to_onehot_list(turn)]  # e.g., [0,1,0] for 'right'
        return fb, 1, turn

    def _record_node_visit(self, node_id: int) -> None:
        """Bumps visit count on the exp object (if available) and in a local map."""
        try:
            self._visit_counts[node_id] += 1
            # also write onto Experience object if the field exists/should exist
            emap = self.memory_graph.experience_map
            for e in getattr(emap, "exps", []):
                if getattr(e, "id", None) == node_id:
                    setattr(e, "visit_count", int(getattr(e, "visit_count", 0)) + 1)
                    break
        except Exception:
            pass

    def _visits(self, node_id: int) -> int:
        try:
            return int(self._visit_counts.get(node_id, 0))
        except Exception:
            return 0

    # ---------------------------------------------------------------------
    # Task-specific PCFG (target room by COLOR)
    # ---------------------------------------------------------------------
    def build_task_pcfg_from_memory(self, mission_color: str, debug: bool = True) -> PCFG:
        """
        Build a PCFG that targets the NEAREST REACHABLE node whose exp.room_color
        matches mission_color. Structure is kept identical to build_pcfg_from_memory:
        NAVPLAN -> PATH_goal ; PATH_goal -> 'STEP_u_v' ... terminals.

        Side-effects (same as your default PCFG builder):
        - self.link_blacklist: set of undirected edges permanently blacklisted
        - self.goal_node_id: chosen goal node id (first candidate)
        - self.graph_used_for_pcfg: adjacency actually used
        """
        mg   = self.memory_graph
        emap = mg.experience_map
        start = mg.get_current_exp_id()

        if not hasattr(self, "link_blacklist") or self.link_blacklist is None:
            self.link_blacklist = set()

        # ---------- B) nodes ----------
        exps = list(getattr(emap, "exps", []))
        if len(exps) == 0:
            raise RuntimeError("No experiences in experience_map")
        id2exp = {e.id: e for e in exps}

        # ---------- C) adjacency (respect link confidence + inferred links) ----------
        graph = self._pcfg_build_adjacency_with_confidence_and_inferred(
            exps=exps,
            start_id=start,
            blacklist=self.link_blacklist,
            infer_link_max_dist=float(getattr(self, "infer_link_max_dist", 12.4)),
            debug=debug
        )
        self.graph_used_for_pcfg = graph
        if debug:
            print("[TASK PCFG] adjacency:", {k: sorted(v) for k, v in graph.items()})

        # ---------- D) candidate goals by color (task logic) ----------
        target_color = str(mission_color).upper().strip()
        # initial candidates for the requested color
        cand_ids = [
            e.id for e in exps
            if getattr(e, "place_kind", None) == "ROOM"
            and getattr(e, "room_color", None) is not None
            and str(getattr(e, "room_color")).upper() == target_color
        ]

        # If our CURRENT node is already that color, exclude it so we "choose another one".
        if start in cand_ids:
            cand_ids = [nid for nid in cand_ids if nid != start]

        # If no usable nodes of the requested color, pick a RANDOM different color present in the map.
        if not cand_ids:
            other_colors = sorted({
                str(getattr(e, "room_color")).upper()
                for e in exps
                if getattr(e, "place_kind", None) == "ROOM"
                and getattr(e, "room_color", None) is not None
                and str(getattr(e, "room_color")).upper() != target_color
            })
            if other_colors:
                rng = getattr(self, "_rng", None)
                fallback_color = (rng.choice(other_colors) if rng else random.choice(other_colors))
                if debug:
                    print(f"[TASK PCFG] No usable {target_color}; switching to random fallback color={fallback_color}")
                target_color = fallback_color
                cand_ids = [
                    e.id for e in exps
                    if getattr(e, "place_kind", None) == "ROOM"
                    and getattr(e, "room_color", None) is not None
                    and str(getattr(e, "room_color")).upper() == target_color
                    and e.id != start  # still avoid selecting current node
                ]

        if debug:
            print(f"[TASK PCFG] target_color={target_color} usable_candidates={sorted(cand_ids)} (start={start})")

        # If still none, fall back to default PCFG (keeps behavior consistent with base builder)
        if not cand_ids:
            if debug:
                print("[TASK PCFG] No usable color candidates; falling back to build_pcfg_from_memory().")
            return self.build_pcfg_from_memory(debug=debug)

        # ---------- E) choose FARTHEST reachable candidate ----------
        def _bfs_all_dists(adj: dict[int, list[int]], src: int) -> dict[int, int]:
            d = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj.get(u, ()):
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            return d

        dists = _bfs_all_dists(graph, start)
        reachable = [(nid, dists[nid]) for nid in cand_ids if nid in dists]
        if not reachable:
            if debug:
                print("[TASK PCFG] Color candidates exist but none reachable; falling back to default PCFG.")
            return self.build_pcfg_from_memory(debug=debug)
        goal = max(reachable, key=lambda kv: (kv[1], kv[0]))[0]
        self.goal_node_id = goal
        if debug:
            print(f"[TASK PCFG] chosen goal node={goal}")

        # ---------- F) enumerate k candidate paths to goal (same pattern) ----------
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
                for nb in graph.get(u, ()):
                    if nb not in p:
                        q.append(p + [nb])
            return out

        sp = self._pcfg_shortest_path(graph, start, goal)
        if debug:
            print(f"[TASK PCFG] shortest path edges: {(len(sp)-1) if sp else None}")

        k_paths   = int(getattr(self, "pcfg_k_paths", 26))
        margin    = int(getattr(self, "pcfg_depth_margin", 4))
        if getattr(self, "pcfg_depth_cap", None):
            depth_cap = int(self.pcfg_depth_cap)
        else:
            depth_cap = ((len(sp)-1) + max(margin, 1)) if sp else (len(exps) + max(margin, 1))

        paths = []
        if sp:
            paths.append(sp)
        extras = all_paths(start, goal, k=k_paths, depth=depth_cap)
        for p in extras:
            if not sp or p != sp:
                paths.append(p)
        if debug:
            print(f"[TASK PCFG] depth_cap={depth_cap}  total_paths={len(paths)}")

        if not paths:
            paths = [[start]]

        # ---------- G) grammar identical to your default builder ----------
        rules = defaultdict(list)
        rules["NAVPLAN"].append((f"PATH_{goal}", 1.0))

        for path in paths:
            step_tokens = []
            if start != goal and not self._at_node_exact(start):
                step_tokens.append(f"STEP_{start}_{start}")
            step_tokens += [f"STEP_{u}_{v}" for u, v in zip(path, path[1:])]
            rhs = " ".join(step_tokens) if step_tokens else f"STEP_{start}_{start}"
            rules[f"PATH_{goal}"].append((rhs, 1.0))

        for lhs in list(rules.keys()):
            if lhs.startswith("PATH_"):
                for rhs, _ in rules[lhs]:
                    for tok in rhs.split():
                        if tok not in rules:
                            rules[tok].append((f"'{tok}'", 1.0))

        lines = []
        for lhs, prods in rules.items():
            Z = sum(p for _, p in prods) or 1.0
            for rhs, p in prods:
                lines.append(f"{lhs} -> {rhs} [{p/Z:.6f}]")

        grammar_src = "\n".join(lines)
        if debug:
            print("[TASK PCFG] Final grammar:\n" + grammar_src)

        return PCFG.fromstring(grammar_src)

    def _is_empty_seq(self, seq) -> bool:
        """True if seq is None, empty list/tuple, or a tensor with numel()==0."""
        try:
            import torch
            if seq is None:
                return True
            if torch.is_tensor(seq):
                return int(seq.numel()) == 0
            try:
                return len(seq) == 0
            except Exception:
                return False
        except Exception:
            # If torch import fails or anything odd, be conservative.
            try:
                return len(seq) == 0
            except Exception:
                return True

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
            infer_link_max_dist=float(getattr(self, "infer_link_max_dist", 9.4)),
            debug=debug
        )

        # Persist for inspection
        self.graph_used_for_pcfg = graph

        if debug:
            print("[PCFG DEBUG] adjacency (post-confidence+inferred):", {k: sorted(v) for k,v in graph.items()})

        # ---------- D) choose GOAL deterministically via sparse-zone heuristic -----
        goal = self._pcfg_select_goal_node_scored(exps=exps, start_id=start, graph=graph, debug=debug)
        print("WE SELECTED VIA SCORE")
        if goal is None:
            goal = self._pcfg_select_goal_node_via_zones(exps=exps, start_id=start, graph=graph, debug=debug)
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
        if goal is not None: 
            # ─── NEW: derive a phantom push near the chosen GOAL (NAVIGATE-only) ───
            try:
                print("[PCFG DEBUG] phantom WE ARE CREATING A PHANTOM")
    
                print("[PCFG DEBUG] phantom WE ARE  CREATING A PHANTOM")
                self._phantom_goal_pose = self._predict_phantom_pose_using_zones(
                    exps=exps, start_id=start, goal_id=goal, graph=graph, debug=debug
                )
                self._phantom_active = False
                if debug and self._phantom_goal_pose is not None:
                    print(f"[PCFG DEBUG] phantom_goal_pose={self._phantom_goal_pose} (derived from zone gap near GOAL)")
                """else:
                    print("[PCFG DEBUG] phantom WE ARE NOT CREATING A PHANTOM")
                    self._phantom_goal_pose = None
                    self._phantom_active = False"""
            except Exception as e:
                if debug: print(f"[PCFG DEBUG] phantom derivation failed: {e}")
                self._phantom_goal_pose = None
                self._phantom_active = False

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
    def ensure_plan_for_current_mode(self, mission: str | None = None, debug: bool = True) -> bool:
        """
        If we're in a nav-like mode (NAVIGATE or TASK_SOLVING) and have no plan (or on enter),
        build the right PCFG and tokenize it. Returns True iff we now have something to execute.
        """
        mode = str(getattr(self, "current_mode", "NAVIGATE"))
        nav_like = mode in ("NAVIGATE", "TASK_SOLVING")

        if not nav_like:
            return False

        need = bool(
            getattr(self, "_need_replan_on_enter", False) or
            (float(getattr(self, "plan_progress", -1.0)) < 0.0) or
            (len(getattr(self, "full_plan_tokens", [])) == 0)
        )
        if not need:
            return True

        # Pick the correct PCFG builder
        if mode == "TASK_SOLVING":
            print("[TASK]are we entering? ")
            if mission is None:
                mission = self.task_solve_mission()
            color = self._parse_color_from_mission(mission)
            if not color:
                if debug:
                    print(f"[ensure_plan] TASK_SOLVING: could not parse color from {mission!r} → default PCFG")
                grammar = self.build_pcfg_from_memory(debug=debug)
            else:
                grammar = self.build_task_pcfg_from_memory(color, debug=debug)
        else:
            grammar = self.build_pcfg_from_memory(debug=debug)

        self.generate_plan_with_pcfg(grammar, debug=debug)
        self._need_replan_on_enter = False

        if debug:
            print(f"[ensure_plan] mode={mode} tokens={self.full_plan_tokens} "
                f"target={self.target_node_id} phantom={self._phantom_goal_pose}")
        return bool(self.full_plan_tokens) or (self._phantom_goal_pose is not None)

    def _safe_log_mode_event(self, kind: str, **kw):
        pe = getattr(self, "plan_export", None)
        if pe is None:
            return
        try:
            pe.log_event(kind, **kw)
        except Exception:
            pass

    def _on_mode_transition(self, new_mode: str, debug=True) -> None:
        
        prev = getattr(self, "_last_mode", None)
        if debug:
            print(f"[mode] {prev} → {new_mode} | has_plan={len(getattr(self,'full_plan_tokens',[]))>0} "
                f"tok={getattr(self,'token_idx',0)}/{len(getattr(self,'full_plan_tokens',[]))} "
                f"prog={float(getattr(self,'plan_progress',-1.0)):.3f}")
        if prev == new_mode:
            return

        NAV_MODES = {"NAVIGATE", "TASK_SOLVING"}
        leaving_nav  = (prev in NAV_MODES) and (new_mode not in NAV_MODES)
        entering_nav = (new_mode in NAV_MODES) and (prev not in NAV_MODES)
        switching_nav_kind = (prev in NAV_MODES) and (new_mode in NAV_MODES) and (prev != new_mode)

        # NEW: treat NAVIGATE and TASK_SOLVING as equivalent for plan continuity
        nav_swap_equiv = switching_nav_kind and getattr(self, "treat_nav_modes_as_equivalent", True)


        
        # ---- Leaving any nav-like mode (unchanged) ----
        if leaving_nav:
            recently_finished = self.navigation_flags.get("plan_complete", False) or \
                                (float(getattr(self, "plan_progress", -1.0)) >= 1.0)
            if recently_finished:
                self._suppress_replan_on_enter = True
                self._suppress_replan_ticks = int(getattr(self, "plan_finished_hold_ticks", 2))
                self._reset_token_prims(); self.prim_idx = 0
                self._evade_injected = False; self._evade_ticks_since_recalc = 0
            else:
                self.full_plan_tokens = []
                self.token_idx = 0
                self.plan_progress = -1.0
                self._reset_token_prims()
                self.prim_idx = 0
                self._phantom_goal_pose = None
                self._phantom_active = False
                self._phantom_attempts = 0

        # ---- Entering any nav-like mode ----
        # Reset volatile low-level state on ANY nav entry OR nav-kind swap (old behavior)
        if entering_nav or switching_nav_kind:
            self._reset_token_prims()
            self.prim_idx = 0
            self._evade_injected = False
            self._evade_ticks_since_recalc = 0

        if entering_nav:
            
            finished = self.navigation_flags.get("plan_complete", False) or \
                    (float(getattr(self, "plan_progress", -1.0)) >= 1.0)
            self._safe_log_mode_event(
                "mode_enter_nav",
                prev=str(prev), new=str(new_mode),
                finished_on_entry=bool(finished),
                prog=float(getattr(self, "plan_progress", -1.0))
            )
            if finished:
                self._post_finish_hold = 0
                self._suppress_replan_on_enter = False
                self._suppress_replan_ticks = 0
                self.navigation_flags.pop("plan_complete", None)
                self.full_plan_tokens = []
                self.token_idx = 0
                self.plan_progress = -1.0
                self._phantom_goal_pose = None
                self._phantom_active = False
                self._phantom_attempts = 0
                self._need_replan_on_enter = True
            else:
                if getattr(self, "_suppress_replan_on_enter", False) and getattr(self, "_suppress_replan_ticks", 0) > 0:
                    self._suppress_replan_ticks -= 1
                    if self._suppress_replan_ticks <= 0:
                        self._suppress_replan_on_enter = False
                    self._need_replan_on_enter = False
                else:
                    self._need_replan_on_enter = True

        # ---- NAV↔TASK swap behavior ----
        if nav_swap_equiv:
            # Unified finished check
            prog = float(getattr(self, "plan_progress", -1.0))
            has_plan = len(getattr(self, "full_plan_tokens", [])) > 0
            finished_tokens = has_plan and (getattr(self, "token_idx", 0) >= len(self.full_plan_tokens))
            finished_flag = bool(self.navigation_flags.get("plan_complete", False))
            finished = finished_flag or finished_tokens or (0.0 <= prog >= 1.0)
            preserved = (has_plan and not finished and (0.0 <= prog < 1.0))
            if preserved:
                self._need_replan_on_enter = False
                if debug:
                    print(f"[mode] NAV swap preserve plan: {prev}->{new_mode} "
                        f"tok={self.token_idx}/{len(self.full_plan_tokens)} prog={prog:.3f}")

            if has_plan and not finished and (0.0 <= prog < 1.0):
                # UNFINISHED plan → PRESERVE (this is the only “equivalence” change)
                self._need_replan_on_enter = False
                if debug:
                    print(f"[mode] NAV swap preserve plan: {prev}->{new_mode} "
                        f"tok={self.token_idx}/{len(self.full_plan_tokens)} prog={prog:.3f}")
            else:
                # FINISHED or NO PLAN → match OLD behavior: BLANK and replan
                self.full_plan_tokens = []
                self.token_idx = 0
                self.plan_progress = -1.0
                self._phantom_goal_pose = None
                self._phantom_active = False
                self._phantom_attempts = 0
                self._need_replan_on_enter = True

            self._safe_log_mode_event(
                "mode_swap_nav_family",
                prev=str(prev), new=str(new_mode),
                preserved=bool(preserved),
                tok=int(getattr(self, "token_idx", 0)),
                n_tokens=int(len(getattr(self, "full_plan_tokens", []))),
                prog=float(getattr(self, "plan_progress", -1.0))
            )

        elif switching_nav_kind and not entering_nav:
            # Original behavior for non-equivalent nav kinds
            self._suppress_replan_on_enter = False
            self._suppress_replan_ticks = 0
            if self.navigation_flags.get("plan_complete", False) or float(getattr(self, "plan_progress", -1.0)) >= 1.0:
                self.full_plan_tokens = []
                self.token_idx = 0
                self.plan_progress = -1.0
            self._need_replan_on_enter = True

        self._last_mode = new_mode




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
        treat_all = bool(getattr(self, "pcfg_treat_all_links_confident", False))
        no_infer  = bool(getattr(self, "pcfg_disable_inferred_links", False))
        ign_bl    = bool(getattr(self, "pcfg_ignore_blacklist", False))

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
                    if not getattr(self, "pcfg_ignore_portal_closures", False) and self._edge_hits_any_closed_portal(e.id, v):
                        dropped += 1  # treat as dropped by geometry
                    else:
                        graph[e.id].append(v)
                else:
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
            # Add inferred edges (undirected → add both directions) with portal-closure check
            kept_inferred_pairs = set()
            for pair in inferred_pairs:
                if (not ign_bl) and (pair in blacklist):
                    continue
                u, v = tuple(pair)

                # safety: still respect blacklist if we're not ignoring it
                if (not ign_bl) and (canon(u, v) in blacklist):
                    continue

                # NEW: drop any inferred edge that crosses a closed portal
                if not getattr(self, "pcfg_ignore_portal_closures", False) and self._edge_hits_any_closed_portal(u, v):
                    continue

                graph[u].append(v)
                graph[v].append(u)
                kept_inferred_pairs.add(pair)

            added_inferred = len(kept_inferred_pairs)

            # ── expose only the kept inferred pairs for exporters/visualization
            try:
                self._pcfg_inferred_pairs = sorted(list(kept_inferred_pairs))
            except Exception:
                self._pcfg_inferred_pairs = []

            # optional: keep the "longest" debug using the kept set
            if debug or dbg_infer:
                longest = sorted(
                    [(math.hypot(pts[u][0]-pts[v][0], pts[u][1]-pts[v][1]), u, v)
                    for (u, v) in [tuple(p) for p in kept_inferred_pairs]],
                    reverse=True
                )[:5]
                if longest:
                    print("[PCFG DEBUG] inferred (longest first) [dist,u,v]:", longest)

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

    def _pcfg_select_goal_node_scored(self, exps, start_id: int, graph: dict[int, list[int]], debug: bool=False):
        """
        Endgame-friendly goal scoring.
        Keeps 'distance kill' if you want it, but fixes crowding and
        augments interest with graph-degree frontier and kNN isolation.
        """
        import math, numpy as np

        # --- Pull nodes (robust xy) ---
        nodes = [(int(e.id),
                float(getattr(e, "x_m", getattr(e, "x", 0.0))),
                float(getattr(e, "y_m", getattr(e, "y", 0.0))))
                for e in exps]
        if len(nodes) < 4:
            return None

        xs = [x for _, x, _ in nodes]; ys = [y for _, _, y in nodes]
        min_x, max_x = min(xs), max(xs);  min_y, max_y = min(ys), max(ys)
        if not all(map(math.isfinite, (min_x, max_x, min_y, max_y))):
            return None
        if (max_x - min_x) < 1e-9 or (max_y - min_y) < 1e-9:
            return None

        # --- Coarse grid for coverage/crowding, like before (4x4) ---
        Nx = 4; Ny = 4
        step_x = (max_x - min_x) / Nx;  step_y = (max_y - min_y) / Ny
        def cell_of(x, y):
            ix = min(Nx-1, max(0, int((x - min_x) / max(step_x, 1e-9))))
            iy = min(Ny-1, max(0, int((y - min_y) / max(step_y, 1e-9))))
            return ix, iy

        counts = [[0 for _ in range(Ny)] for _ in range(Nx)]
        by_cell = {}
        for nid, x, y in nodes:
            ij = cell_of(x, y)
            counts[ij[0]][ij[1]] += 1
            by_cell.setdefault(ij, []).append(nid)

        # --- Coverage metrics (fill_frac, imbalance) ---
        flat = [counts[i][j] for i in range(Nx) for j in range(Ny)]
        nonzero = [c for c in flat if c > 0]
        nonempty_cells = sum(1 for c in flat if c > 0)
        fill_frac = nonempty_cells / max(len(flat), 1)

        def _gini(arr):
            n = len(arr); s = sum(arr)
            if n == 0 or s == 0: return 0.0
            arr_sorted = sorted(arr)
            cum = sum((i+1)*v for i, v in enumerate(arr_sorted))
            return (2.0*cum)/(n*s) - (n+1)/n

        imbalance = _gini(flat)

        # --- Per-cell interest ingredients (coarse grid) ---
        # 1) scarcity
        scarcity_cell = [[1.0/(1.0 + counts[i][j]) for j in range(Ny)] for i in range(Nx)]
        # 2) deficit vs median of NONZERO cells
        med_nz = max((sorted(nonzero)[len(nonzero)//2] if nonzero else 0), 1)
        deficit_cell = [[max(0.0, (med_nz - counts[i][j]) / med_nz) for j in range(Ny)] for i in range(Nx)]
        # 3) coarse frontier: fraction of empty neighbors in 3×3 ring
        empty = {(i,j) for i in range(Nx) for j in range(Ny) if counts[i][j] == 0}
        def frontier_frac(ix, iy):
            tot = emp = 0
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if dx == 0 and dy == 0: continue
                    x2, y2 = ix+dx, iy+dy
                    if 0 <= x2 < Nx and 0 <= y2 < Ny:
                        tot += 1
                        if (x2, y2) in empty: emp += 1
            return (emp/tot) if tot > 0 else 0.0
        frontier_cell = [[frontier_frac(i, j) for j in range(Ny)] for i in range(Nx)]
        # --- Border awareness & interiorness ---
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        map_rad = 0.5 * math.hypot(max_x - min_x, max_y - min_y)

        def border_ring(ix, iy, Nx, Ny):
            # 1.0 for outer ring, 0.5 for second ring, 0 inside
            bx = 1.0 if ix in (0, Nx-1) else (0.5 if ix in (1, Nx-2) else 0.0)
            by = 1.0 if iy in (0, Ny-1) else (0.5 if iy in (1, Ny-2) else 0.0)
            return max(bx, by)

        beta_border = float(getattr(self, "nav_w_border_dampen", 0.35 if fill_frac >= 0.75 else 0.20))
        gamma_interior = float(getattr(self, "nav_w_interior_pull", 0.25 if fill_frac >= 0.75 else 0.10))

        # --- Graph-degree frontier & kNN isolation (per-node) ---
        id_set = {nid for nid, _, _ in nodes}
        deg = {nid: len([v for v in graph.get(nid, ()) if v in id_set]) for nid,_,_ in nodes}
        deg_max = max(deg.values()) if deg else 1
        # normalize so leaves (deg=0/1) ≈1.0, hubs ≈0.0
        def deg_frontier_of(nid):
            return 1.0 - min(1.0, deg[nid] / max(1.0, deg_max))

        pts = {nid: (x, y) for nid, x, y in nodes}
        iso_k = int(getattr(self, "nav_iso_k", 2))
        iso_vals = {}
        for nid, (xi, yi) in pts.items():
            dists = [math.hypot(xi - xj, yi - yj)
                    for mj, (xj, yj) in pts.items() if mj != nid]
            dists.sort()
            if len(dists) == 0:
                iso_vals[nid] = 0.0
            else:
                m = np.mean(dists[:min(iso_k, len(dists))])
                iso_vals[nid] = m
        iso_med = float(np.median(list(iso_vals.values()) or [0.0])) or 1.0
        def iso_norm_of(nid):
            # >1 ⇒ very isolated; clip to [0,1] against median for robustness
            return float(np.clip(iso_vals[nid] / max(iso_med, 1e-6), 0.0, 1.0))

        # --- Interest mix (tilt weights in endgame) ---
        if fill_frac >= float(getattr(self, "nav_endgame_fill_frac", 0.75)):
            a_def, a_scar, a_front = 0.25, 0.25, 0.50  # push to frontier late
            b_deg  = float(getattr(self, "nav_w_deg_frontier", 0.25))
            b_iso  = float(getattr(self, "nav_w_iso",          0.25))
        else:
            a_def, a_scar, a_front = 0.50, 0.35, 0.15
            b_deg  = float(getattr(self, "nav_w_deg_frontier", 0.10))
            b_iso  = float(getattr(self, "nav_w_iso",          0.10))

        # --- BFS (still needed for ties / early stage) ---
        dists = self._pcfg_bfs_all_dists(graph, start_id)

        # --- Base weights ---
        w_interest = float(getattr(self, "nav_goal_w_interest", 1.5))
        w_dist     = float(getattr(self, "nav_goal_w_dist",     0.35))
        w_visit    = float(getattr(self, "nav_goal_w_visit",    0.65))

        # distance scaling (you want hard kill late, keep it)
        a_fill = float(getattr(self, "nav_goal_dist_decay_fill", 1.0))
        a_imb  = float(getattr(self, "nav_goal_dist_decay_imb",  0.6))
        floor  = float(getattr(self, "nav_goal_w_dist_floor",    0.05))
        s = 1.0 - (a_fill * fill_frac + a_imb * imbalance)
        s = max(floor / max(w_dist, 1e-6), min(1.0, s))
        if fill_frac >= float(getattr(self, "nav_endgame_fill_frac", 0.75)) and \
        bool(getattr(self, "nav_endgame_kill_dist", True)):
            s = 0.0
        w_dist_eff = w_dist * max(0.0, s)

        # visit lookup
        id2exp = {int(e.id): e for e in exps}

        # crowding (percentiles) on the coarse grid
        q50 = float(np.quantile(flat, 0.50)) if flat else 0.0
        q90 = float(np.quantile(flat, 0.90)) if flat else 1.0
        den = max(q90 - q50, 1.0)
        w_crowd = float(getattr(self, "nav_goal_w_crowd", 0.35))
        if fill_frac >= 0.75:
            w_crowd = float(getattr(self, "nav_goal_w_crowd_endgame", 0.6))

        # --- Score candidates ---
        hard_limit = int(getattr(self, "goal_visit_hard_limit", 2))
        scored = []
        for nid, x, y in nodes:
            if nid == start_id:        continue
            if nid not in dists:       continue

            # respect hard visit cut if you truly want it
            v_seen = int(getattr(self, "_visit_counts", {}).get(nid, 0))
            v_exp  = int(getattr(id2exp.get(nid, object()), "visit_count", 0))
            v = max(v_seen, v_exp)
            if v >= hard_limit:        continue

            ix, iy = cell_of(x, y)

            # cell-level interest
            base_cell_interest = (
                a_def   * deficit_cell[ix][iy] +
                a_scar  * scarcity_cell[ix][iy] +
                a_front * frontier_cell[ix][iy]
            )

            # node-level augmentations
            deg_front = deg_frontier_of(nid)    # [0..1], high = leaf/bridge
            iso_norm  = iso_norm_of(nid)        # [0..1], high = isolated in (x,y)
            # Border dampening only when the coarse cell is NOT a frontier (we don't want to mute real frontiers)
            ring = border_ring(ix, iy, Nx, Ny)
            front_local = frontier_cell[ix][iy]  # ∈ [0,1]
            border_dampen = beta_border * ring * (1.0 - front_local)

            # Interior pull (only a gentle tilt; 1 at center → more weight to interior isolation)
            r_norm = math.hypot(x - cx, y - cy) / max(map_rad, 1e-6)      # 0 center … 1 hull
            interiorness = float(np.clip(1.0 - r_norm, 0.0, 1.0))

            # Apply to node-level signals
            deg_front_eff = deg_front * (1.0 - border_dampen)
            
            iso_eff = iso_norm * (1.0 - border_dampen) * (1.0 + gamma_interior * interiorness)

            # Combine with cell interest
            s_interest = (1.0 - (b_deg + b_iso)) * base_cell_interest + b_deg * deg_front_eff + b_iso * iso_eff
            s_interest = float(np.clip(s_interest, 0.0, 1.0))

            # crowd penalty (percentile-normalized)
            c = counts[ix][iy]
            crowd_norm = 0.0 if c <= q50 else (c - q50) / den

            hops = float(dists[nid])
            s_visit = math.log1p(float(v))
            if fill_frac >= 0.75:
                w_crowd = float(getattr(self, "nav_goal_w_crowd_endgame", 0.7))  # was 0.6

            score = (w_interest * s_interest) - (w_dist_eff * hops) - (w_visit * s_visit) - (w_crowd * crowd_norm)

            # tie-break: prefer lower crowd, then fewer hops (even if distance is killed, small bias),
            # then higher id for determinism
            if debug:
                def dbg_row(nid, sc):
                    print(f" nid={nid:>3}  score={sc:.3f}  s_int={s_interest:.3f}  "
                        f"degF={deg_front_eff:.2f} iso={iso_eff:.2f}  "
                        f"crowd={crowd_norm:.2f}  hops={hops:.0f}  visits={v}")
            scored.append((score, -crowd_norm, -hops, -nid, nid))

        if not scored:
            if debug: print("[PCFG SCORE] no scored candidates; returning None")
            return None

        scored.sort(reverse=True)
        best = scored[0][-1]
        if debug:
            top5 = [(nid, f"{sc:.3f}", f"crowd={-cr:.2f}", f"hops={-hp}") for (sc, cr, hp, _nidneg, nid) in scored[:5]]
            print(f"[PCFG SCORE] fill_frac={fill_frac:.2f}  imbalance(Gini)={imbalance:.2f}  w_dist_eff={w_dist_eff:.3f}")
            print(f"[PCFG SCORE] picked goal={best}  top5={top5}")
        return best




    def _pcfg_select_goal_node_via_zones(
        self,
        exps,
        start_id: int,
        graph: dict[int, list[int]] | None,
        debug: bool = False,
    ) -> int | None:
        """
        Frontier-biased goal selection:
        1) 4×4 (or dynamic) partition of node bbox.
        2) Prefer EMPTY cells; else bottom-quartile sparse cells.
        3) Score candidate cells by emptiness, distance from start, and corner bias.
        4) From top cell(s), pick nodes that are (a) near the cell centroid,
            (b) locally sparse (big kNN radius), (c) outward (near bbox edge),
            (d) far in BFS hops (if graph provided).
        Returns best node id or None.
        """

        if not exps or len(exps) < 4:
            return None

        # Pull nodes as (id, x, y, kind)
        def _xy(e):
            return float(getattr(e, "x_m", getattr(e, "x", 0.0))), float(getattr(e, "y_m", getattr(e, "y", 0.0)))
        nodes = [(int(e.id), *_xy(e), str(getattr(e, "place_kind", ""))) for e in exps]

        # Start pose
        try:
            sx, sy, _ = getattr(self.memory_graph.experience_map, "get_pose")(start_id)
        except Exception:
            # Fallback: coordinates of the start node
            sx, sy = next(((x, y) for nid, x, y, _ in nodes if nid == start_id), (nodes[0][1], nodes[0][2]))

        xs = [x for _, x, _, _ in nodes]
        ys = [y for _, _, y, _ in nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if not (math.isfinite(min_x) and math.isfinite(max_x) and math.isfinite(min_y) and math.isfinite(max_y)):
            return None
        if abs(max_x - min_x) < 1e-6 or abs(max_y - min_y) < 1e-6:
            return None

        # Grid size: keep 4×4 by default; you can make it dynamic if you like.
        Nx = Ny = 4
        step_x = (max_x - min_x) / Nx
        step_y = (max_y - min_y) / Ny

        def cell_of(x, y):
            ix = min(Nx - 1, int((x - min_x) / max(step_x, 1e-9)))
            iy = min(Ny - 1, int((y - min_y) / max(step_y, 1e-9)))
            return ix, iy

        cell_counts = [[0 for _ in range(Ny)] for _ in range(Nx)]
        for _, x, y, _ in nodes:
            ix, iy = cell_of(x, y)
            cell_counts[ix][iy] += 1

        # Empty cells first, else bottom-quartile sparse
        all_cells = [(ix, iy) for ix in range(Nx) for iy in range(Ny)]
        empties = [(ix, iy) for (ix, iy) in all_cells if cell_counts[ix][iy] == 0]
        counts_flat = sorted(c for col in cell_counts for c in col)
        q1 = counts_flat[len(counts_flat) // 4] if counts_flat else 0
        sparse = [(ix, iy) for (ix, iy) in all_cells if 0 < cell_counts[ix][iy] <= q1]
        cand_cells = empties if len(empties) > 0 else sparse
        if not cand_cells:
            # Degenerate fallback: farthest node by BFS then Euclid
            if graph:
                dists = self._pcfg_bfs_all_dists(graph, start_id) or {}
                if dists:
                    return max(dists.items(), key=lambda kv: (kv[1], kv[0]))[0]
            # Pure Euclid fallback
            return max(nodes, key=lambda t: (math.hypot(t[1] - sx, t[2] - sy), -t[0]))[0]

        def centroid(ix, iy):
            return min_x + (ix + 0.5) * step_x, min_y + (iy + 0.5) * step_y

        # Cell scoring: emptiness (1.0 for empty), far from start, corner/outside bias
        max_count = max([max(col) for col in cell_counts]) or 1
        # normalize distance range
        dists_centroids = []
        for c in cand_cells:
            cx, cy = centroid(*c)
            dists_centroids.append(math.hypot(cx - sx, cy - sy))
        dmin = min(dists_centroids) if dists_centroids else 0.0
        dmax = max(dists_centroids) if dists_centroids else 1.0

        def cell_score(ix, iy):
            cx, cy = centroid(ix, iy)
            dist = math.hypot(cx - sx, cy - sy)
            far = 0.0 if dmax <= dmin else (dist - dmin) / (dmax - dmin)
            empty_bonus = 1.0 if cell_counts[ix][iy] == 0 else 1.0 - (cell_counts[ix][iy] / max_count)
            corner_bonus = 1.0 if (ix in (0, Nx - 1) and iy in (0, Ny - 1)) else (0.5 if (ix in (0, Nx - 1) or iy in (0, Ny - 1)) else 0.0)
            # weights: tweakable
            return 0.5 * empty_bonus + 0.3 * far + 0.2 * corner_bonus

        cand_cells.sort(key=lambda ij: cell_score(*ij), reverse=True)
        best_cell = cand_cells[0]
        cx, cy = centroid(*best_cell)
        radius = 0.75 * max(step_x, step_y)  # consider nodes near the empty/sparse cell

        # Precompute kNN distances (k=3) for local sparsity
        coords = [(nid, x, y) for (nid, x, y, _) in nodes]
        def kNN_radius(nid, x, y, k=3):
            ds = []
            for (mid, mx, my) in coords:
                if mid == nid: 
                    continue
                ds.append((mx - x) ** 2 + (my - y) ** 2)
            if len(ds) == 0:
                return 0.0
            ds.sort()
            k = min(k, len(ds))
            # mean of k nearest as a "sparsity" proxy
            return sum(ds[:k]) / k

        # Optional BFS (graph can be None)
        bfs_d = {}
        if graph:
            try:
                bfs_d = self._pcfg_bfs_all_dists(graph, start_id) or {}
            except Exception:
                bfs_d = {}
        max_bfs = max(bfs_d.values()) if bfs_d else 1

        # Candidate nodes: those within radius of the target cell centroid (or top-10 nearest)
        cands = []
        near = []
        for nid, x, y, kind in nodes:
            d2c = (x - cx) ** 2 + (y - cy) ** 2
            near.append((d2c, nid, x, y, kind))
        near.sort()
        for tup in near:
            d2c, nid, x, y, kind = tup
            if len(cands) < 10 or math.sqrt(d2c) <= radius:
                cands.append((nid, x, y, kind, d2c))
            else:
                break
        if not cands:
            # Shouldn't happen, but be safe
            cands = [(nid, x, y, kind, (x - cx) ** 2 + (y - cy) ** 2) for (nid, x, y, kind) in nodes]

        # Score nodes: outwardness + local sparsity + BFS hops + closeness to target cell
        def outwardness(x, y):
            # smaller edge distance => more outward
            edge_d = min(x - min_x, max_x - x, y - min_y, max_y - y)
            rng = max(max_x - min_x, max_y - min_y) or 1.0
            return 1.0 - max(0.0, min(1.0, edge_d / (0.5 * rng)))  # in [0,1], 1=more outward

        best_node = None
        best_score = -1e9
        for nid, x, y, kind, d2c in cands:
            s_out = outwardness(x, y)
            s_knn = kNN_radius(nid, x, y, k=3)  # bigger = sparser
            s_knn_norm = s_knn
            s_bfs = (bfs_d.get(nid, 0) / max_bfs) if bfs_d else 0.0
            s_close = 1.0 / (1.0 + d2c)  # stay near the empty/sparse cell

            # weights (tune to taste) — DOOR term removed
            score = (0.30 * s_out) + (0.25 * s_knn_norm) + (0.15 * s_bfs) + (0.10 * s_close)

            if nid == start_id and len(nodes) > 1:
                score -= 1.0

            if score > best_score or (score == best_score and (best_node is None or nid < best_node)):
                best_score = score
                best_node = nid

        if best_node is None and bfs_d:
            
            best_node = max(bfs_d.items(), key=lambda kv: (kv[1], kv[0]))[0]
            print("[NODE SELECTION]fallback",best_node)
        return best_node

    def _flush_pending_node_media(self):
        try:
            pend = getattr(self, "_pending_node_media", None)
            if not pend:
                return
            for nid, snap, enhanced in pend:
                try:
                    self.plan_export.node_created(
                        node_id=int(nid),
                        snapshot_img=snap if snap is not None else None,
                        enhanced_pred_imgs=enhanced if (enhanced and len(enhanced) > 0) else None,
                    )
                except Exception as e:
                    try:
                        self.plan_export.log_event("node_media_flush_error", node_id=int(nid), err=str(e))
                    except Exception:
                        pass
            self._pending_node_media = []
        except Exception:
            pass
    def generate_plan_with_pcfg(self, grammar: PCFG, debug: bool = True) -> list[tuple[int, int]]:
        """
        Deterministic: choose the SHORTEST PATH_* production (fewest STEP edges) and tokenize it.
        This respects the grammar structure you build:
        NAVPLAN -> PATH_goal [1.0]
        PATH_goal -> STEP_u_v STEP_v_w ...  (multiple alternatives)
        We ignore any 'STEP_u_u' warmup/self-edge when counting cost.
        """
        from nltk.grammar import Nonterminal

        def _is_step_sym(sym) -> bool:
            if isinstance(sym, Nonterminal):
                return str(sym.symbol()).startswith("STEP_")
            return isinstance(sym, str) and sym.startswith("STEP_")

        def _tok_of(sym) -> str | None:
            if isinstance(sym, Nonterminal):
                return str(sym.symbol()) if str(sym.symbol()).startswith("STEP_") else None
            return sym if isinstance(sym, str) and sym.startswith("STEP_") else None

        def _edge_of(tok: str) -> tuple[int, int] | None:
            # tok: "STEP_u_v"
            try:
                _, a, b = tok.split("_")
                return int(a), int(b)
            except Exception:
                return None

        def _path_cost(prod) -> int:
            # Count only real edges (ignore STEP_u_u warmups)
            cost = 0
            for sym in prod.rhs():
                tok = _tok_of(sym)
                if tok:
                    e = _edge_of(tok)
                    if e is not None and e[0] != e[1]:
                        cost += 1
            return cost

        # --- 1) Find the PATH_* nonterminal from NAVPLAN (should be exactly one) ---
        nav_nt = Nonterminal("NAVPLAN")
        nav_prods = grammar.productions(lhs=nav_nt)
        if not nav_prods:
            raise ValueError("PCFG has no NAVPLAN productions.")

        # Pick the highest-prob (your builder makes this 1.0 anyway)
        nav_best = max(nav_prods, key=lambda p: p.prob())

        # Extract PATH_* symbol on RHS
        path_nts = [sym for sym in nav_best.rhs()
                    if isinstance(sym, Nonterminal) and str(sym.symbol()).startswith("PATH_")]
        if not path_nts:
            # Fallback: look for any PATH_* LHS in the whole grammar
            all_path_lhss = [p.lhs() for p in grammar.productions()
                            if str(p.lhs()).startswith("PATH_")]
            if not all_path_lhss:
                raise ValueError("PCFG has no PATH_* nonterminals.")
            path_nt = all_path_lhss[0]
        else:
            path_nt = path_nts[0]

        # --- 2) Among PATH_goal alternatives, choose the shortest (fewest edges) ---
        path_prods = grammar.productions(lhs=path_nt)
        if not path_prods:
            raise ValueError(f"No productions for {path_nt} in PCFG.")

        # Tie-break by fewer total RHS symbols for determinism, then by textual rhs
        best_prod = min(
            path_prods,
            key=lambda p: (_path_cost(p), len(p.rhs()), " ".join(str(s) for s in p.rhs()))
        )

        # --- 3) Tokenize the chosen RHS into [(u,v), ...] ---
        edges: list[tuple[int, int]] = []
        for sym in best_prod.rhs():
            tok = _tok_of(sym)
            if not tok:
                continue
            e = _edge_of(tok)
            if e is None:
                continue
            u, v = e
            if u == v:
                # Ignore self-edge warmups in the executable token list
                continue
            edges.append((u, v))

        if debug:
            step_names = [f"STEP_{u}_{v}" for (u, v) in edges]
            print(f"[PCFG FASTEST] chosen {path_nt} with {len(edges)} edges")
            print(f"[PCFG FASTEST] steps: {' '.join(step_names)}")

        # --- 4) Bookkeeping identical to your previous function ---
        self.full_plan_tokens = edges
        # ── Exporter: begin run with full plan snapshot
        try:
            mg = self.memory_graph
            graph = getattr(self, "graph_used_for_pcfg", {}) or {}
            inferred = getattr(self, "_pcfg_inferred_pairs", [])
            start_id = mg.get_current_exp_id() if hasattr(mg, "get_current_exp_id") else None
            self.plan_export.begin(
                mode=str(getattr(self, "current_mode", "NAVIGATE")),
                mg=mg, graph=graph, inferred_pairs=inferred,
                tokens=self.full_plan_tokens,
                goal_id=getattr(self, "goal_node_id", None),
                start_id=start_id,
                mode_session_idx=int(getattr(self, "_mode_session_idx", 1)),  # ← NEW
            )
            # NEW: drain any early-captured media now that run_dir exists
            self._flush_pending_node_media()

        except Exception as _e:
            if debug: print(f"[export] begin_run failed: {_e}")

        self.token_idx = self.prim_idx = 0
        self._reset_token_prims()
        self.plan_progress = 0.0

        if self.full_plan_tokens:
            _, self.target_node_id = self.full_plan_tokens[-1]
        else:
            self.target_node_id = None

        if debug:
            print(f"[PCFG FASTEST] Target node ID: {self.target_node_id}")

        return self.full_plan_tokens

        
    def _tokenize_plan(self, full_plan: str) -> list[tuple[int, int]]:
        """Convert 'STEP_9_8 STEP_8_5 …' → [(9,8), (8,5), …]."""
        return [
            tuple(map(int, t.split('_')[1:3]))
            for t in full_plan.strip().split()
            if t.startswith('STEP_')
        ]
    


    def _finish_plan(self, debug=False):
        # mark finished in a way universal_navigation can reliably detect
        print("we are finishing the plan")
        self.plan_progress = 1.0
        self.navigation_flags['plan_complete'] = True

        # post-finish hold: give outer controller a couple of ticks to consume the flag
        self._post_finish_hold = int(getattr(self, "plan_finished_hold_ticks", 2))

        # drop volatile low-level state
        self._reset_token_prims()
        self.prim_idx = 0
        self._evade_injected = False
        # clear per-edge/partial bookkeeping so a future plan starts clean
        self._partial_attempts.clear()
        self._edge_key = None
        self._edge_evades = 0
        # ---- NEW: arm once when a TASK_SOLVING run finishes ----
        try:
            if str(getattr(self, "current_mode", "")) == "TASK_SOLVING":
                # first completion only; change to > =0 if you want to arm every time
                if (not self.task_mutation_armed) and (int(getattr(self, "task_runs_completed", 0)) == 0):
                    self.task_runs_completed = 1
                    self.task_mutation_armed = True
                    # mirror to flags in case you prefer polling flags
                    self.navigation_flags["task_mutation_armed"] = True
                    if debug: print("[TASK] armed task_mutation (first completion)")
        except Exception:
            pass

    def consume_task_mutation_armed(self) -> bool:
        armed = bool(self.task_mutation_armed or self.navigation_flags.get("task_mutation_armed", False))
        if armed:
            self.task_mutation_armed = False
            self.navigation_flags.pop("task_mutation_armed", None)
        return armed
    def _plan_finished_now(self) -> bool:
        """True if this tick the plan is finished and we should NOT replan or emit fallback turns."""
        if getattr(self, "_phantom_active", False):
            return False
        if self.navigation_flags.get("plan_complete", False):
            return True
        try:
            return float(self.plan_progress) >= 1.0
        except Exception:
            return False
    def step_plan(self, wm, belief, debug: bool = True):
        """
        Realtime executor: compute A* fresh EACH STEP and emit exactly ONE primitive.
        """
        import torch
        
        # ── Exporter: detect node arrival when token_idx advanced since last call
        try:
            if int(getattr(self, "token_idx", 0)) > int(getattr(self, "_export_last_token_idx", -1)):
                k = int(self.token_idx)
                if k > 0 and k <= len(getattr(self, "full_plan_tokens", [])):
                    _, v = self.full_plan_tokens[k - 1]
                    pose = self.get_current_pose() if callable(getattr(self, "get_current_pose", None)) else None
                    get_fov = getattr(self, "get_fov_image", None)  # user sets: nav_system.get_fov_image = lambda: <H×W×3 uint8>
                    self.plan_export.node_reached(v, pose, get_fov)
            self._export_last_token_idx = int(self.token_idx)
        except Exception as _e:
            if debug: print(f"[export] node_reached hook err: {_e}")

        # ---------- helpers ----------
        def _is_empty(seq) -> bool:
            try:
                if seq is None: return True
                if torch.is_tensor(seq): return int(seq.numel()) == 0
                return len(seq) == 0
            except Exception:
                return True

        def _first_prim(path):
            """Return (onehot:list[3], label:str) for the FIRST primitive in `path`."""
            if _is_empty(path):
                return None, None
            step0 = path[0] if not torch.is_tensor(path) else (path[0] if path.ndim >= 1 else None)
            if step0 is None:
                return None, None

            if torch.is_tensor(step0):
                v = step0.reshape(-1)
                if v.numel() != 3: return None, None
                idx = int(v.argmax().item())
                return v.cpu().tolist(), ("forward", "right", "left")[idx]

            if isinstance(step0, (list, tuple)) and len(step0) == 3:
                try:
                    import numpy as _np
                    idx = int(_np.argmax(step0))
                except Exception:
                    idx = max(range(3), key=lambda i: step0[i])
                return list(step0), ("forward", "right", "left")[idx]

            if isinstance(step0, str) and step0 in ("forward", "left", "right"):
                return self.to_onehot_list(step0), step0

            return None, None

        def _plan_to_node(tgt_id):
            pose = self.get_current_pose()
            if pose is None:
                if debug: print("[step_plan_rt] no current pose")
                return [], {"reached_goal": False, "best_dist": None}
            start = self._pose_to_state(pose)
            path, meta = self._astar_to_node_partial(
                wm, belief, start, tgt_id, debug=debug,
                max_partial_len=getattr(self, "max_partial_len", 8)
            )
            # normalize
            if path is None or (torch.is_tensor(path) and path.numel() == 0) or (hasattr(path, "__len__") and len(path) == 0):
                path = []
            return path, (meta if isinstance(meta, dict) else {"reached_goal": True, "best_dist": None})

        def _plan_to_pose(goal_pose_xyz):
            pose = self.get_current_pose()
            if pose is None:
                if debug: print("[step_plan_rt] no current pose for phantom")
                return [], {"reached_goal": False, "best_dist": None}
            start = self._pose_to_state(pose)
            path, meta = self._astar_to_pose_partial(
                wm, belief, start, goal_pose_xyz, debug=debug,
                max_partial_len=int(getattr(self, "phantom_partial_len", 6))
            )
            if path is None or (torch.is_tensor(path) and path.numel() == 0) or (hasattr(path, "__len__") and len(path) == 0):
                path = []
            return path, (meta if isinstance(meta, dict) else {"reached_goal": True, "best_dist": None})

        def _finish_done():
            self._finish_plan(debug=debug)
            return [], 0

        # ---------- skip edges whose v we are already at ----------
        while self.token_idx < len(self.full_plan_tokens):
            u, v = self.full_plan_tokens[self.token_idx]
            if self._at_node_exact(v):
                if debug: print(f"[step_plan_rt] at target v={v} → record+advance")
                self._record_node_visit(v)
                self.token_idx += 1
                continue
            break

        # ---------- phantom when no more edges ----------
        if self.token_idx >= len(self.full_plan_tokens):
            if self._phantom_goal_pose is not None:
                path, meta = _plan_to_pose(self._phantom_goal_pose)
                try:
                    self.plan_export.log_event(
                        "phantom_step",
                        reached=bool(meta.get('reached_goal', True)),
                        path_len=(int(path.shape[0]) if hasattr(path, "shape") else (len(path) if hasattr(path,"__len__") else 0))
                    )
                except Exception: pass
                # >>> add: track phantom partial retries
                if not meta.get('reached_goal', True):
                    self._phantom_attempts = int(getattr(self, "_phantom_attempts", 0)) + 1
                else:
                    self._phantom_attempts = 0
                
                if self._phantom_attempts >= int(getattr(self, "phantom_retries_before_finish", 6)):
                    if debug: print("[step_plan_rt] phantom retries exhausted → finish")
                    try: self.plan_export.log_event("phantom_finish", reason="retries_exhausted")
                    except Exception: pass
                    self._phantom_goal_pose = None
                    self._phantom_active = False
                    self._phantom_attempts = 0
                    return _finish_done()
                if _is_empty(path):
                    if debug: print("[step_plan_rt] phantom empty → finish")
                    try: self.plan_export.log_event("phantom_finish", reason="empty_path")
                    except Exception: pass
                    self._phantom_goal_pose = None
                    self._phantom_active = False
                    return _finish_done()

                prim_out, prim_lab = _first_prim(path)
                if prim_out is None:
                    if debug: print("[step_plan_rt] phantom decode fail → finish")
                    self._phantom_goal_pose = None
                    self._phantom_active = False
                    return _finish_done()

                # grading bookkeeping
                self._phantom_active = True
                self._issue_seq = int(getattr(self, "_issue_seq", 0)) + 1
                hist = getattr(self, "_pose_hist", ())
                self._issue_baseline_hist_len = len(hist)
                cur = self.get_current_pose()
                if cur is not None:
                    self._issue_baseline_pose = (float(cur[0]), float(cur[1]), float(cur[2]))
                    self._issue_baseline_valid = True
                else:
                    self._issue_baseline_valid = False
                self._last_served_prim = prim_lab

                # IMPORTANT: while phantom is active, treat plan as "not yet finished"
                # so upstream won't prematurely short-circuit.
                self.navigation_flags.pop('plan_complete', None)
                self.plan_progress = min(0.99, self.token_idx / max(len(self.full_plan_tokens), 1))
                try:
                    get_fov = getattr(self, "get_fov_image", None)
                    img = get_fov() if callable(get_fov) else None
                    pose = self.get_current_pose() if callable(getattr(self, "get_current_pose", None)) else None
                    self.plan_export.step_obs(edge=None, token_idx=int(getattr(self, "token_idx", -1)),
                                              pose=pose, action=prim_lab, image=img)
                except Exception as _e:
                    if debug: print(f"[export] step_obs (phantom) err: {_e}")

                return [prim_out], 1

            if debug: print("[step_plan_rt] plan finished (no tokens, no phantom)")
            return _finish_done()

        # ---------- realtime A* to current edge target v ----------
        u, v = self.full_plan_tokens[self.token_idx]
        path, meta = _plan_to_node(v)

        if _is_empty(path):
            if debug: print(f"[step_plan_rt] EMPTY A* to v={v} → request replan")
            try:
                key = (u, v)
                cur_attempts = int(getattr(self, "_partial_attempts", {}).get(key, 0)) + 1
                self.plan_export.log_event("astar_empty", edge=[int(u), int(v)], attempts=cur_attempts)
            except Exception: pass
            self.plan_progress = -1.0
            key = (u, v)
            self._partial_attempts[key] += 1

            if self._partial_attempts[key] >= int(getattr(self, "partial_retries_before_replan", 8)):
                self.navigation_flags['replan_request'] = True
                self.navigation_flags['replan_bad_node'] = v
                try:
                    self.plan_export.log_event("replan_request_armed",
                        reason="empty_path", edge=[int(u), int(v)],
                        attempts=int(self._partial_attempts[key]))
                except Exception: pass
                self._partial_attempts[key] = 0
            return [], 0

        # partial accounting
        key = (u, v)
        if not meta.get('reached_goal', True):
            self._partial_attempts[key] += 1
            if debug: print(f"[step_plan_rt] partial towards v={v} attempts={self._partial_attempts[key]}")
            try:
                self.plan_export.log_event("partial_attempt",
                    edge=[int(u), int(v)],
                    attempts=int(self._partial_attempts[key]),
                    best_dist=float(meta.get('best_dist', -1.0)) if isinstance(meta, dict) else None
                )
            except Exception: pass
            if self._partial_attempts[key] >= int(getattr(self, "partial_retries_before_replan", 3)):
                if debug: print("[step_plan_rt] partial limit → request replan")
                self.navigation_flags['replan_request'] = True
                self.navigation_flags['replan_bad_node'] = v
                try:
                    self.plan_export.log_event("replan_request_armed",
                        reason="partial_limit", edge=[int(u), int(v)],
                        attempts=int(self._partial_attempts[key]))
                except Exception: pass
                
                self._partial_attempts[key] = 0
        else:
            try: self._partial_attempts.pop(key, None)
            except Exception: pass

        prim_out, prim_lab = _first_prim(path)
        
        if prim_out is None:
            if debug: print("[step_plan_rt] first-prim decode failed")
            self.plan_progress = -1.0
            return [], 0

        # grading bookkeeping
        self._issue_seq = int(getattr(self, "_issue_seq", 0)) + 1
        hist = getattr(self, "_pose_hist", ())
        self._issue_baseline_hist_len = len(hist)
        cur = self.get_current_pose()
        if cur is not None:
            self._issue_baseline_pose = (float(cur[0]), float(cur[1]), float(cur[2]))
            self._issue_baseline_valid = True
        else:
            self._issue_baseline_valid = False
        self._last_served_prim = prim_lab

        self.plan_progress = self.token_idx / max(len(self.full_plan_tokens), 1)
        # ── EXPORT: per-step FoV for current edge (u→v)
        try:
            get_fov = getattr(self, "get_fov_image", None)
            img = get_fov() if callable(get_fov) else None
            pose = self.get_current_pose() if callable(getattr(self, "get_current_pose", None)) else None
            ek = (u, v)
            self._edge_key = ek  # keep around for detour-injected logs, if you use it elsewhere
            self.plan_export.step_obs(edge=ek, token_idx=int(getattr(self, "token_idx", -1)),
                                      pose=pose, action=prim_lab, image=img)
        except Exception as _e:
            if debug: print(f"[export] step_obs err: {_e}")
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
    def _edge_hits_any_closed_portal(self, a, b) -> bool:
        self._ensure_portal_store()
        if not self._closed_portals:
            return False
        emap = self.memory_graph.experience_map
        (_, _), (_, _), mid_ab, dir_ab = _edge_features(emap, a, b)
        import math
        # normalize once
        d = math.hypot(dir_ab[0], dir_ab[1]) + 1e-9
        dir_ab = (dir_ab[0]/d, dir_ab[1]/d)
        for P in self._closed_portals:
            ang = _angdiff(P["dir"], dir_ab)
            if ang > float(getattr(self, "portal_angle_tol_deg", 25.0)):
                continue
            dx, dy = (mid_ab[0]-P["mid"][0], mid_ab[1]-P["mid"][1])
            perp = (-P["dir"][1], P["dir"][0])
            lon = abs(dx*P["dir"][0] + dy*P["dir"][1])
            lat = abs(dx*perp[0]   + dy*perp[1])
            if (lon <= P["half_len"]) and (lat <= P["half_w"]):
                return True
        return False
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

    def _astar_to_node_partial(self, wm, belief, start: "State", v: int,
                           debug: bool = False,
                           max_partial_len: int | None = None):
        if max_partial_len is None:
            max_partial_len = getattr(self, "max_partial_len", 8)

        # Goal as State
        gx, gy, gd = self.memory_graph.experience_map.get_pose(v)
        goal_state = State(int(round(gx)), int(round(gy)), int(gd))

        try:
            # Ask planner for partials without extra kwargs
            ret = self.planner.astar_prims(
                wm, belief, start, goal_state,
                verbose=debug,
                allow_partial=True
            )
            if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
                path, meta = ret
            else:
                path, meta = ret, {'reached_goal': True, 'best_dist': None}
        except TypeError:
            print("no FULL PARTIAL")
            # Planner doesn't support allow_partial yet → legacy full path
            path = self.planner.astar_prims(wm, belief, start, goal_state, verbose=debug)
            meta = {'reached_goal': True, 'best_dist': None}

        # Normalize empties and cap partial length on the caller side
        try:
            import torch
            if path is None:
                path = []
            if torch.is_tensor(path) and path.numel() == 0:
                path = []
        except Exception:
            if path is None:
                path = []

        # If planner returned a partial and capping is desired, slice here
        if not meta.get('reached_goal', True):
            try:
                # tensor [T,3]
                if hasattr(path, "shape") and len(path.shape) == 2 and path.shape[0] > max_partial_len:
                    path = path[:max_partial_len]
                # list
                elif hasattr(path, "__len__") and len(path) > max_partial_len:
                    path = path[:max_partial_len]
            except Exception:
                pass

        return path, meta

    def _highlevel_replan_from_here(self, penalize_pair, penalize_node, dbg=print) -> bool:
        """Extracted so we can call consistently from several places."""
        prev_tokens = list(getattr(self, "full_plan_tokens", []))
        try:
            print("[blacklist] do we even have penalize pair?",penalize_pair)
            if penalize_pair is not None:
                u, v = int(penalize_pair[0]), int(penalize_pair[1])
                # blacklist both directions
                link = self._find_link(u, v)
                if link is not None: link.confidence = 0
                
                if not hasattr(self, "link_blacklist") or self.link_blacklist is None:
                    self.link_blacklist = set()
                print("are we actually blackilisting?",u,v,int(penalize_pair[0]), int(penalize_pair[1]))
                self.link_blacklist.add(frozenset({u, v}))
                
                self._register_portal_closure(u, v, dbg=dbg)
        except Exception as e:
            dbg(f"[rt] warn: link penalty failed: {e}")

        try:
            dbg("[rt] High-level REPLAN via PCFG…")
            mode = str(getattr(self, "current_mode", "NAVIGATE"))
            if mode == "TASK_SOLVING":
                # Let the mode-aware ensure pick the right PCFG (task color vs default)
                ok_before = len(self.full_plan_tokens) > 0
                self.ensure_plan_for_current_mode(debug=getattr(self, "debug_universal_navigation", False))
                ok = len(self.full_plan_tokens) > 0 or (self._phantom_goal_pose is not None)
            else:
                grammar = self.build_pcfg_from_memory(debug=getattr(self, "debug_universal_navigation", False))
                self.generate_plan_with_pcfg(grammar, debug=getattr(self, "debug_universal_navigation", False))
                ok = len(self.full_plan_tokens) > 0 or (self._phantom_goal_pose is not None)

            # Reset low-level flags regardless:
            self.navigation_flags.pop('replan_request', None)
            self.navigation_flags.pop('replan_bad_node', None)
            self.navigation_flags.pop('stalled_motion', None)
            self.plan_progress = 0.0
            
            try:
                prev = prev_tokens
                new  = getattr(self, "full_plan_tokens", [])
                self.plan_export.bump_plan_version(
                    prev_tokens=prev,
                    new_tokens=new,
                    reason=reason or {"kind": "replan"},  # 'reason' dict you already built above
                )
            except Exception:
                pass


            try:
                reason = {}
                if penalize_pair is not None:
                    reason["penalize_pair"] = [int(penalize_pair[0]), int(penalize_pair[1])]
                if penalize_node is not None:
                    reason["penalize_node"] = int(penalize_node)
                self.plan_export.log_event(
                    "replan",
                    success=bool(ok),
                    reason=reason,
                    new_tokens=[list(map(int,t)) for t in getattr(self, "full_plan_tokens", [])],
                )
            except Exception:
                pass

            dbg(f"[rt] REPLAN mode={mode} success={ok} tokens={self.full_plan_tokens} phantom={self._phantom_goal_pose}")
            return ok
        except Exception as e:
            dbg(f"[rt] High-level replan failed: {e}")
            return False


    def set_mode(self, new_mode: str):
        self.current_mode = new_mode
        self._on_mode_transition(new_mode)
    def universal_navigation(self, submode: str, wm, belief) -> tuple[list[str], int]:
        """
        Controller policy:
        1) Handle 'stalled_motion' / 'replan_request' up front.
        2) Always try to emit exactly ONE primitive via step_plan() (serves phantom too).
        3) If the plan is finished:
            - respect the brief post-finish hold, but DO NOT emit a fake action
            - then YIELD ([],0) so outer modes can switch.
        4) If not finished and we couldn't emit, try ONE high-level replan.
            If that still emits nothing and we aren't finished, last resort is a turn.
            If the replan ends up finished, yield.
        Key change: finished -> NEVER emit a fallback turn. Yield control cleanly.
        """
        dbg = getattr(self, "debug_universal_navigation", False)
        def _log(msg: str):
            if dbg: print(msg)

        # --- helpers -----------------------------------------------------------
        def _yield_noop() -> tuple[list[str], int]:
            # Do NOT bump _issue_seq; this tells navigation_grade there is
            # no new primitive and lets the outer controller switch modes.
            _log("[navigate] yielding control ([],0)")
            return [], 0
        
        def _stall_turn() -> tuple[list[str], int]:
            # Only use when we explicitly want to stay in NAVIGATE while not finished.
            import random
            turn = random.choice(["right", "left"])
            self._last_served_prim = turn

            # Issue accounting so navigation_grade sees a primitive (NOT used on finish)
            self._issue_seq = int(getattr(self, "_issue_seq", 0)) + 1
            hist = getattr(self, "_pose_hist", ())
            self._issue_baseline_hist_len = len(hist)
            if len(hist) > 0:
                self._issue_baseline_pose = hist[-1]
                self._issue_baseline_valid = True
            else:
                cur = self.get_current_pose()
                if cur is not None:
                    self._issue_baseline_pose = (float(cur[0]), float(cur[1]), float(cur[2]))
                    self._issue_baseline_valid = True
                else:
                    self._issue_baseline_valid = False

            fb = [self.to_onehot_list(turn)]
            _log(f"[navigate] stall-turn → issuing {fb}")
            return fb, 1

        # keep your transition hygiene
        mode_here = str(getattr(self, "current_mode", "NAVIGATE"))
        # Transition hygiene with the actual mode we are in
        self._on_mode_transition(mode_here)
        # Ensure there is a plan when entering any nav-like mode (NAVIGATE or TASK_SOLVING)
        if mode_here in ("NAVIGATE", "TASK_SOLVING"):
            if getattr(self, "_need_replan_on_enter", False):
                # Only skip replan if we truly have an *active* (unfinished) plan
                has_plan = len(self.full_plan_tokens) > 0
                prog     = float(getattr(self, "plan_progress", -1.0))
                finished = self.navigation_flags.get("plan_complete", False) or (prog >= 1.0)

                if has_plan and (0.0 <= prog < 1.0) and not finished:
                    self._need_replan_on_enter = False
                else:
                    self.ensure_plan_for_current_mode(debug=dbg)
            elif (float(getattr(self, "plan_progress", -1.0)) < 0.0 or
                len(getattr(self, "full_plan_tokens", [])) == 0):
                # Don’t replan if we *just* finished and the higher level hasn’t switched modes yet.
                if not self.navigation_flags.get("plan_complete", False):
                    self.ensure_plan_for_current_mode(debug=dbg)
       

        _log(f"[universal_navigation_rt] submode={submode} pose={self.get_current_pose()} "
            f"tok={self.token_idx}/{len(self.full_plan_tokens)} prog={self.plan_progress:.3f}")
        
        # --- 0) React to stalled/replan requests BEFORE trying to step ----------
        if self.navigation_flags.get('stalled_motion', False):
            _log("[rt] stalled_motion → REPLAN")
            cur_edge = self.full_plan_tokens[self.token_idx] if self.token_idx < len(self.full_plan_tokens) else None
            try:
                cur_edge = self.full_plan_tokens[self.token_idx] if self.token_idx < len(self.full_plan_tokens) else None
                self.plan_export.log_event("replan_requested", source="stalled_motion",
                                        edge=(list(cur_edge) if cur_edge else None),
                                        token_idx=int(self.token_idx))
            except Exception: pass
            ok = self._highlevel_replan_from_here(cur_edge, cur_edge[1] if cur_edge else None, dbg=_log)
            self.navigation_flags.pop('stalled_motion', None)
            if not ok:
                # Do NOT emit a fake action; let higher layers handle the stall.
                return _yield_noop()

        if self.navigation_flags.get('replan_request', False):
            bad_node = self.navigation_flags.get('replan_bad_node', None)
            cur_edge = self.full_plan_tokens[self.token_idx] if self.token_idx < len(self.full_plan_tokens) else None
            _log(f"[rt] replan_request → penalize {cur_edge} bad_node={bad_node}")
            try:
                cur_edge = self.full_plan_tokens[self.token_idx] if self.token_idx < len(self.full_plan_tokens) else None
                bad_node = self.navigation_flags.get('replan_bad_node', None)
                self.plan_export.log_event("replan_requested", source="partial_limit_or_empty",
                                        edge=(list(cur_edge) if cur_edge else None),
                                        bad_node=(int(bad_node) if bad_node is not None else None),
                                        token_idx=int(self.token_idx))
            except Exception: pass
            ok = self._highlevel_replan_from_here(cur_edge, bad_node, dbg=_log)
            if not ok:
                return _yield_noop()

        # --- 1) Try to emit ONE primitive (step_plan serves phantom too) -------
        prims, n = self.step_plan(wm, belief, debug=dbg)
        if prims and n:
            return prims, n

        # --- 2) If nothing emitted, check if we're finished/handing off -------
        finished = bool(self.navigation_flags.get('plan_complete', False)) or \
                (float(getattr(self, "plan_progress", -1.0)) >= 1.0)

        if finished:
            # Phantom would have been served already by step_plan.
            # Respect a brief post-finish hold (for any grading consumers),
            # but do NOT emit fake actions; then yield control.
            if getattr(self, "_post_finish_hold", 0) > 0:
                self._post_finish_hold -= 1
                return _yield_noop()
            _log("[rt] plan finished & hold expired → yield control")
            try: self.plan_export.log_event("plan_finished_yield")
            except Exception: pass
            return _yield_noop()

        # --- 3) Not finished, no prims → attempt ONE high-level replan --------
        _log("[rt] no prims and not finished → attempt one REPLAN")
        ok = self._highlevel_replan_from_here(None, None, dbg=_log)
        if ok:
            prims, n = self.step_plan(wm, belief, debug=dbg)
            if prims and n:
                return prims, n

            # Re-check finished after replan
            finished = bool(self.navigation_flags.get('plan_complete', False)) or \
                    (float(getattr(self, "plan_progress", -1.0)) >= 1.0)
            if finished:
                if getattr(self, "_post_finish_hold", 0) > 0:
                    self._post_finish_hold -= 1
                    return _yield_noop()
                _log("[rt] finished after replan & hold expired → yield control")
                return _yield_noop()

        # --- 4) Last resort while not finished: a single stall turn -----------
        try:
            self.plan_export.log_event("stall_turn", reason="last_resort", token_idx=int(getattr(self, "token_idx", -1)))
        except Exception: pass
        return _stall_turn()

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
        if mode not in ('NAVIGATE', 'TASK_SOLVING'):
            if hasattr(self, "_edge_evades"): self._edge_evades = 0
            self.navigation_flags.pop('evade_request', None)
            self.navigation_flags.pop('replan_request', None)
            self.navigation_flags.pop('replan_bad_node', None)
            _log("mode not in NAV_MODES → 0.0")
            return 0.0

        if self.token_idx >= len(self.full_plan_tokens):
            if getattr(self, "_phantom_active", False):
                # Keep grading as if plan is active
                pass
            else:
                _log(f"no tokens to grade: token_idx={self.token_idx} len={len(self.full_plan_tokens)} → -1.0")
                return -1.0
    
        if self.navigation_flags.pop('synthetic_action', False):
            _log("synthetic external action → 0.0")
            return 0.0
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
