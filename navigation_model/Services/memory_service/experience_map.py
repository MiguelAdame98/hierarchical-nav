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
import math
from operator import itemgetter
from .modules import *
from sys import maxsize
import math
import heapq
import torch
import cv2
# ---------- SIMPLE THETA DEBUG (toggle on/off here) ----------
EM_THETA_DEBUG = True     # set False to silence
EPS_TH = 1e-6
NO_DRIFT_MODE = True  

def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def theta_snap(em, tag=""):
    """One-liner snapshot of angles and deltas."""
    if not EM_THETA_DEBUG: return
    if getattr(em, "current_exp", None) is not None:
        ax, ay, aθ = em.current_exp.x_m, em.current_exp.y_m, em.current_exp.facing_rad
        gx = ax + em.accum_delta_x
        gy = ay + em.accum_delta_y
    else:
        ax = ay = 0.0; aθ = 0.0
        gx = em.accum_delta_x; gy = em.accum_delta_y
    dθ  = em.accum_delta_facing
    θw  = _wrap_pi(aθ + dθ)
    print(f"[θ][{tag}] a={aθ:+.3f} Δ={dθ:+.3f} w={θw:+.3f} "
          f"Δxy=({em.accum_delta_x:+.3f},{em.accum_delta_y:+.3f}) "
          f"GP=({gx:+.3f},{gy:+.3f})")

def theta_reanchor_check(em, tag=""):
    """After (re)anchor, deltas must be exactly zero; world θ must equal anchor θ."""
    if not EM_THETA_DEBUG: return
    ok_xy = abs(em.accum_delta_x) <= EPS_TH and abs(em.accum_delta_y) <= EPS_TH
    ok_dθ = abs(_wrap_pi(em.accum_delta_facing)) <= EPS_TH
    if getattr(em, "current_exp", None) is not None:
        θw = _wrap_pi(em.current_exp.facing_rad + em.accum_delta_facing)
        ok_θ = abs(_wrap_pi(θw - em.current_exp.facing_rad)) <= EPS_TH
    else:
        ok_θ = True
    ok = ok_xy and ok_dθ and ok_θ
    print(f"[θ][{tag}] REANCHOR {'OK' if ok else 'PROBLEM'} "
          f"Δxy=({em.accum_delta_x:+.3f},{em.accum_delta_y:+.3f}) Δθ={em.accum_delta_facing:+.3f}")

def theta_forward_check(em, vtrans: float, prev_dx: float, prev_dy: float, tag=""):
    """For a pure translation step, moved direction must align with world θ."""
    if not EM_THETA_DEBUG or abs(vtrans) < EPS_TH: return
    aθ = em.current_exp.facing_rad if getattr(em, "current_exp", None) is not None else 0.0
    θw = _wrap_pi(aθ + em.accum_delta_facing)
    dx = em.accum_delta_x - prev_dx
    dy = em.accum_delta_y - prev_dy
    # project onto heading and lateral
    fwd =  dx*np.cos(θw) + dy*np.sin(θw)
    lat = -dx*np.sin(θw) + dy*np.cos(θw)
    ok  = (fwd > 0.5*vtrans) and (abs(lat) < 1e-3)
    print(f"[θ][{tag}] MOVE {'OK' if ok else 'PROBLEM'} "
          f"stepΔ=({dx:+.3f},{dy:+.3f}) proj_fwd={fwd:+.3f} lateral={lat:+.3f}")

def theta_link_check(u, v, link, tag=""):
    """Right after creating a link: check heading_rad & facing_rad geometry."""
    if not EM_THETA_DEBUG: return
    dx, dy = (v.x_m - u.x_m), (v.y_m - u.y_m)
    abs_head = np.arctan2(dy, dx)
    head_exp = _wrap_pi(abs_head - u.facing_rad)      # expected link.heading_rad
    face_exp = _wrap_pi(v.facing_rad - u.facing_rad)  # expected link.facing_rad
    dh = abs(_wrap_pi(link.heading_rad - head_exp))
    df = abs(_wrap_pi(link.facing_rad  - face_exp))
    ok = (dh <= 1e-6 and df <= 1e-6)
    print(f"[θ][{tag}] LINK {u.id}->{v.id} {'OK' if ok else 'PROBLEM'} "
          f"h(exp={head_exp:+.3f},link={link.heading_rad:+.3f},Δ={dh:.1e}) "
          f"f(exp={face_exp:+.3f},link={link.facing_rad:+.3f},Δ={df:.1e})")

def theta_gp_check(em, tag=""):
    """Compare get_global_position() vs anchor+deltas recompute."""
    if not EM_THETA_DEBUG: return
    gx, gy, gθ = em.get_global_position()
    if getattr(em, "current_exp", None) is not None:
        ax, ay, aθ = em.current_exp.x_m, em.current_exp.y_m, em.current_exp.facing_rad
    else:
        ax = ay = aθ = 0.0
    x2 = ax + em.accum_delta_x
    y2 = ay + em.accum_delta_y
    θ2 = _wrap_pi(aθ + em.accum_delta_facing)
    ok = (abs(gx - x2) <= 1e-6 and abs(gy - y2) <= 1e-6 and abs(_wrap_pi(gθ - θ2)) <= 1e-6)
    print(f"[θ][{tag}] GP {'OK' if ok else 'PROBLEM'} "
          f"get=({gx:+.3f},{gy:+.3f},{gθ:+.3f}) "
          f"re=({x2:+.3f},{y2:+.3f},{θ2:+.3f})")
# ---------- END SIMPLE THETA DEBUG ----------

from collections import namedtuple
State = namedtuple("State", ["x","y","d"])
DIR_VECS = [(1,0),(0,1),(-1,0),(0,-1)]
ACTIONS   = ["forward","left","right"]

class Experience(object):
    '''A single experience.'''

    _ID = 0
    _ghost_ID = 0

    def __init__(self,
                 x_pc, y_pc, th_pc,
                 x_m, y_m, facing_rad,
                 view_cell, local_pose, ghost_exp,
                 real_pose=None,
                 pose_cell_pose=None):
        '''Initializes the Experience.'''
        # Basic pose-cell and map coordinates
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.x_m = x_m
        self.y_m = y_m
        self.facing_rad = facing_rad

        # Cell references
        self.view_cell = view_cell
        self.init_local_position = local_pose

        # New pose attributes
        #self.imagined_pose= imagined_pose
        self.real_pose= real_pose
        self.pose_cell_pose= [x_pc, y_pc, th_pc]

        self.confidence=1
        # Link list
        self.links = []

        # ID assignment
        self.ghost_exp = ghost_exp
        if self.ghost_exp:
            self.id = Experience._ghost_ID
            Experience._ghost_ID += 1
        else:
            self.id = Experience._ID
            Experience._ID += 1
        self.place_kind = None     # "ROOM", "CORRIDOR", or "UNKNOWN"
        self.room_color = None     # e.g., "red", "purple", or None
        self.grid_xy    = None     # (gx, gy) in env grid coord space


        print(f"[DEBUG][Experience __init__] Created {'ghost' if ghost_exp else ''}Exp{self.id}: ")
        print(f"   pose_cell=({self.x_pc},{self.y_pc},{self.th_pc}), ")
        print(f"   map=({self.x_m:.3f},{self.y_m:.3f},{self.facing_rad:.3f}), ")
        print(f"   real_pose={self.real_pose}, pose_cell_pose={self.pose_cell_pose}")

    def link_to(self, target,
                accum_delta_x, accum_delta_y, accum_delta_facing,
                active_link,
                confidence=1):
        """
        Create a directed link self → target and append it to self.links.
        Debug prints included.
        """
        print(f"\n[DEBUG][link_to] from Exp{self.id} to Exp{target.id}")
        print(f"   deltas: dx={accum_delta_x:.3f}, dy={accum_delta_y:.3f}, dtheta={accum_delta_facing:.3f}")
        print(f"   self.map=({self.x_m:.3f},{self.y_m:.3f},{self.facing_rad:.3f}), target.map=({target.x_m:.3f},{target.y_m:.3f},{target.facing_rad:.3f})")
        if target is self:
            print(f"[WARN][link_to] ignoring self-link request {self.id}->{target.id}")
            return None
        # geometry
        d = np.hypot(accum_delta_x, accum_delta_y)
        abs_heading = np.arctan2(accum_delta_y, accum_delta_x)
        heading_rad = signed_delta_rad(self.facing_rad, abs_heading)
        facing_rad  = signed_delta_rad(self.facing_rad, accum_delta_facing)

        print(f"   computed abs_heading={abs_heading:.3f}, heading_rad={heading_rad:.3f}, d={d:.3f}, facing_rad={facing_rad:.3f}")

        link = ExperienceLink(
            parent=self,
            target=target,
            facing_rad=facing_rad,
            d=d,
            heading_rad=heading_rad,
            active_link=active_link,
            confidence=int(confidence),
            
        )
        self.links.append(link)
        print(f"   [DEBUG] Stored link: {link}")

    def __repr__(self):
        return f"[Exp {self.id}]"

    def update_link(self, link, e1):
        print(f"[DEBUG][update_link] Exp{self.id} updating link to Exp{e1.id}")
        if link.active_link:
            delta_x = e1.x_m - self.x_m
            delta_y = e1.y_m - self.y_m
        else:
            delta_x = self.x_m - e1.x_m
            delta_y = self.y_m - e1.y_m
        delta_facing = signed_delta_rad(self.facing_rad, e1.facing_rad)
        print(f"   deltas: dx={delta_x:.3f}, dy={delta_y:.3f}, dtheta={delta_facing:.3f}")

        link.d = np.hypot(delta_x, delta_y)
        link.heading_rad = signed_delta_rad(self.facing_rad, np.arctan2(delta_y, delta_x))
        link.facing_rad = delta_facing
        print(f"   updated link: {link}")
        return link


class ExperienceLink(object):
    '''Connection between two experiences, with debug prints.'''

    def __init__(self, parent, target,
                 facing_rad, d, heading_rad,
                 active_link,
                 confidence=1):
        self.parent       = parent
        self.target       = target
        self.facing_rad   = facing_rad
        self.d            = d
        self.heading_rad  = heading_rad
        self.active_link  = active_link
        self.confidence   = int(confidence) 
        # Store primitive paths
        

        
        print(f"[DEBUG][ExperienceLink __init__] Created link {self.parent.id}->{self.target.id}", self.confidence)
   

    def __repr__(self):
        return (f"ExperienceLink({self.parent.id}->{self.target.id}, "
                f"conf={self.confidence}, d={self.d:.2f}, "
                f"heading={self.heading_rad:.2f}, active={self.active_link})")


class ExperienceMap(object):
    '''Experience Map module with debug prints.'''

    def __init__(self, dim_xy=61, dim_th=36,
                 delta_exp_threshold=1.0, delta_pc_threshold=1.0,
                 correction=0.5, loops=100,
                 constant_adjust=False,
                 replay_buffer=None,
                 
                 **kwargs):
        print("[DEBUG][ExperienceMap __init__] initializing with replay_buffer=", bool(replay_buffer))
        
        self.replay_buffer = replay_buffer
        self._recent_prims: list[str] = []
        self.env = kwargs.get("env", None)
        self.area_room_threshold = kwargs.get("area_room_threshold", 12)
        self.last_link_action= None
        self.last_real_pose= None
        self.last_imagined_pose= None
        # existing fields
        self.DIM_XY = dim_xy
        self.DIM_TH = dim_th
        self.DELTA_EXP_THRESHOLD = delta_exp_threshold
        self.DELTA_PC_THRESHOLD = delta_pc_threshold
        self.CORRECTION = correction
        self.LOOPS = loops
        print("past constant adjust", constant_adjust)
        self.constant_adjust = constant_adjust

        self.size = 0
        self.exps = []
        self.ghost_exps = []
        self.current_exp = None
        self.current_view_cell = None
        self.accum_delta_x = 0
        self.accum_delta_y = 0
        self.accum_delta_facing = 0.0
    def set_env(self, env):
        """Optionally attach the MiniGrid environment so new exps get annotated."""
        self.env = env
    
    def rgb_to_template64(self,img, eps: float = 1e-6, device="cpu"):
        """
        56×56×3 or 64×64×3 RGB  →  64-D descriptor
        • 48 dims = 16-bin histograms of L*, a*, b*
        • 16 dims = 4×4 block-mean edge magnitudes (Sobel)
        Returns: torch.float32 (64,)
        """
        import numpy as np, torch, cv2
        print(img)
        print(type(img))
        # --- 0) to HWC uint8 ---
        if torch.is_tensor(img):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1,3):  # CHW -> HWC
                arr = arr.transpose(1,2,0)
        else:
            arr = np.asarray(img)

        assert arr.ndim == 3 and arr.shape[-1] == 3, f"expected HxWx3, got {arr.shape}"

        # --- 1) resize to 56×56 (the old pipeline assumes 56 for 14×14 blocks) ---
        if arr.shape[0] != 56 or arr.shape[1] != 56:
            arr = cv2.resize(arr, (56,56), interpolation=cv2.INTER_AREA)

        # --- 2) uint8 range ---
        if arr.dtype != np.uint8:
            # assume 0..1 floats or other; clamp to [0,255]
            arr = np.clip(arr, 0, 255)
            if arr.max() <= 1.0 + 1e-6:
                arr = (arr * 255.0).round()
            arr = arr.astype(np.uint8)

        # --- 3) 16-bin Lab histograms (48 dims) ---
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2Lab).astype(np.float32)
        L   = (lab[:, :, 0] * 255.0 / 100.0).clip(0, 255)
        a   = lab[:, :, 1] + 128.0
        b   = lab[:, :, 2] + 128.0

        bins = np.linspace(0, 256, 17, dtype=np.float32)
        h_L, _ = np.histogram(L, bins=bins)
        h_a, _ = np.histogram(a, bins=bins)
        h_b, _ = np.histogram(b, bins=bins)

        h48 = np.concatenate([h_L, h_a, h_b]).astype(np.float32)
        h48 /= h48.sum() + eps

        # --- 4) 4×4 Sobel-edge energy (16 dims) ---
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        mag  = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        edge16 = [
            mag[y:y+14, x:x+14].mean()
            for y in range(0, 56, 14)
            for x in range(0, 56, 14)
        ]
        edge16 = np.asarray(edge16, np.float32)
        edge16 /= edge16.sum() + eps

        # --- 5) concat → 64-D torch.float32 ---
        vec64 = np.concatenate([h48, edge16]).astype(np.float32)
        return torch.from_numpy(vec64).to(device)

    def rgb56_to_template64(self,
        img,
        eps: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        56×56×3 RGB  →  64-D descriptor
            • 48 dims = 16-bin histograms of L*, a*, b*
            • 16 dims = 4×4 block-mean edge magnitudes (Sobel)
        """

        # ------------------------------------------------------------------ #
        # 1. Make sure we have HWC uint8                                     #
        # ------------------------------------------------------------------ #


        if torch.is_tensor(img):
            if img.shape == (3, 56, 56):                      # CHW tensor
                img = img.permute(1, 2, 0).contiguous().cpu().numpy()
                
            else:
                img = img.cpu().numpy()
                
        else:
            if img.shape == (3, 56, 56):                      # CHW numpy
                img = np.transpose(img, (1, 2, 0))
                

        assert img.shape == (56, 56, 3), f"expected 56×56×3, got {img.shape}"

        if img.dtype != np.uint8:
            img = (img * 255.0).round().astype(np.uint8)
            

        # ------------------------------------------------------------------ #
        # 2. 16-bin Lab histograms (48 dims)                                 #
        # ------------------------------------------------------------------ #
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
        L   = (lab[:, :, 0] * 255.0 / 100.0).clip(0, 255)
        a   = lab[:, :, 1] + 128.0
        b   = lab[:, :, 2] + 128.0

        bins = np.linspace(0, 256, 17, dtype=np.float32)
        h_L, _ = np.histogram(L, bins=bins)
        h_a, _ = np.histogram(a, bins=bins)
        h_b, _ = np.histogram(b, bins=bins)

        h48 = np.concatenate([h_L, h_a, h_b]).astype(np.float32)
        h48 /= h48.sum() + eps

        # ------------------------------------------------------------------ #
        # 3. 4×4 Sobel-edge energy (16 dims)                                 #
        # ------------------------------------------------------------------ #
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mag  = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        edge16 = [
            mag[y : y + 14, x : x + 14].mean()
            for y in range(0, 56, 14)
            for x in range(0, 56, 14)
        ]
        edge16 = np.asarray(edge16, np.float32)
        edge16 /= edge16.sum() + eps

        # ------------------------------------------------------------------ #
        # 4. Concatenate → 64-D  & return torch tensor                       #
        # ------------------------------------------------------------------ #
        vec64 = np.concatenate([h48, edge16])
        print("[DBG] vec64  shape", vec64.shape, " first5", vec64[:5])

        return torch.from_numpy(vec64).to(device)


    def _create_exp(self, x_pc, y_pc, th_pc,
                    view_cell, local_pose):
        real     = tuple(self.last_real_pose)

        print("\n[DEBUG][_create_exp] creating new experience")
        self.size += 1
        if self.current_exp is not None:
            gx, gy, gth = self.get_global_position()   # (x_cur + Δx, y_cur + Δy, wrap(f_cur + Δθ))
            x_m, y_m, facing_rad = gx, gy, gth
        else:
            x_m = self.accum_delta_x
            y_m = self.accum_delta_y
            facing_rad = clip_rad_180(self.accum_delta_facing)
        
        facing_rad = self._quantize_cardinal(facing_rad)
        theta_reanchor_check(self, "after-anchor")
        print(f"   posed at cell=({x_pc},{y_pc},{th_pc}), map=({x_m:.3f},{y_m:.3f},{facing_rad:.3f})")
        print(view_cell.id)
        exp = Experience(
            x_pc, y_pc, th_pc,
            x_m, y_m, facing_rad,
            view_cell, local_pose, ghost_exp=False,
            real_pose=real,
            pose_cell_pose=[x_pc,y_pc,th_pc]
        )
        if getattr(self, "env", None) is not None:
            self._annotate_place_from_env(exp, self.env)
        if hasattr(self, "env") and self.env is not None:
            self._annotate_place_from_env(exp, self.env)
        
        e = exp
        print(f"[PLACE] Expppppppp{e.id} at {e.grid_xy}: {e.place_kind}"
            + (f" ({e.room_color})" if e.room_color else ""))
        print(self.get_global_position())
            # ------------------------------------------------------------------
            # Handle *empty* A* result safely
            # ------------------------------------------------------------------
        if self.current_exp is not None:
            self.current_exp.link_to(
                exp,
                self.accum_delta_x,
                self.accum_delta_y,
                exp.facing_rad,
                active_link=True,
                confidence=1
            )
            # backward link
            exp.link_to(
                self.current_exp,
                -self.accum_delta_x,
                -self.accum_delta_y,
                self.current_exp.facing_rad,
                active_link=False,
                confidence=1
            )
            theta_link_check(self.current_exp, exp, self.current_exp.links[-1], "create->fwd")
            theta_link_check(exp, self.current_exp, exp.links[-1], "create->back")
        self.exps.append(exp)
        if view_cell:
            view_cell.exps.append(exp)
        print(f"[DEBUG][_create_exp] total experiences = {len(self.exps)}")
        return exp
    
    
    def get_pose(self, exp_id: int) -> tuple[float, float, float]:
        """
        Return the (x_m, y_m, facing_rad) of the Experience with id==exp_id
        """
        for exp in self.exps:
            if exp.id == exp_id:
                return exp.real_pose
        raise KeyError(f"No experience with id={exp_id}")
    
    def _map_raw_action(self, raw):
        # unchanged
        if isinstance(raw, (list, tuple)) and len(raw)==3:
            idx = max(range(3), key=lambda i: raw[i])
            return ['forward','right','left'][idx]
        if isinstance(raw, int):
            return {0:'left',1:'right',2:'forward'}.get(raw,'forward')
        if isinstance(raw,str):
            low = raw.lower()
            if 'forw' in low: return 'forward'
            if 'left' in low: return 'left'
            if 'right' in low: return 'right'
        return 'forward'

    def _get_recent_prims(self, u_id: int, v_id: int) -> list[str]:
        """
        1) Try to find the exact span where we went u_id → v_id (as before).
        2) If we’ve not yet recorded v_id in the buffer, fall back to
           “all actions performed while in u_id,” so that we capture
           the entire drift inside u_id before the final leave.
        """
        rb = self.replay_buffer or []
        L  = len(rb)
        print(f"[DEBUG3][_get_recent_prims] buffer length = {L}, looking for transition {u_id}->{v_id}")

        # ——— Attempt the exact u→v slice ———
        entry_idx = None
        for i in range(L-1, -1, -1):
            if rb[i].get('node_id') == v_id:
                entry_idx = i
                break

        if entry_idx is not None:
            # (existing logic: scan back to find start_idx, u_idx, etc.)
            start_idx = entry_idx
            while start_idx > 0 and rb[start_idx - 1].get('node_id') == v_id:
                start_idx -= 1

            # find last time we were at u_id
            u_idx = None
            for j in range(start_idx - 1, -1, -1):
                if rb[j].get('node_id') == u_id:
                    u_idx = j
                    break

            action_seq = []
            if u_idx is not None:
                for k in range(u_idx + 1, start_idx + 1):
                    raw  = rb[k].get('action')
                    prim = self._map_raw_action(raw)
                    action_seq.append(prim)
                    print(f"  [DEBUG3] idx {k-1}->{k}: nodes {rb[k-1]['node_id']}->{rb[k]['node_id']},"
                          f" raw_action={raw} -> prim={prim}")
            print(f"  [DEBUG3] collected buffer‐based prims = {action_seq}")
            return action_seq

        # ——— Fallback: we haven’t even seen v_id yet ———
        print(f"  [DEBUG3] never saw v_id={v_id} in buffer; collecting actions while in u_id={u_id}")
        seq = []
        # walk backwards while still in u_id
        idx = L - 1
        while idx >= 0 and rb[idx].get('node_id') == u_id:
            raw  = rb[idx].get('action')
            prim = self._map_raw_action(raw)
            # prepend to build chronological order
            seq.insert(0, prim)
            print(f"  [DEBUG3] idx {idx-1}->{idx}: nodes {rb[idx-1]['node_id'] if idx>0 else '??'}->{u_id},"
                  f" raw_action={raw} -> prim={prim}")
            idx -= 1

        print(f"  [DEBUG3] collected same‐node prims = {seq}")
        return seq


    def update_exp_wt_view_cell(self, updated_exp, x_pc, y_pc, th_pc,
                            new_view_cell=None, local_pose=None):
        # anchor + deltas
        x_m = updated_exp.x_m + self.accum_delta_x
        y_m = updated_exp.y_m + self.accum_delta_y
        facing_rad = clip_rad_180(updated_exp.facing_rad + self.accum_delta_facing)
        facing_rad = self._quantize_cardinal(facing_rad)  # 4 orientations

        updated_exp.x_pc = x_pc
        updated_exp.y_pc = y_pc
        updated_exp.th_pc = th_pc
        updated_exp.x_m = x_m
        updated_exp.y_m = y_m
        updated_exp.facing_rad = facing_rad

        if local_pose is not None:
            updated_exp.init_local_position = local_pose

        if new_view_cell is not None:
            for e in updated_exp.view_cell.exps:
                if e.id == updated_exp.id:
                    updated_exp.view_cell.exps.remove(e)
                    break
            updated_exp.view_cell = new_view_cell
            new_view_cell.exps.append(updated_exp)

        # IMPORTANT: re-anchored → zero the odom deltas
        self.accum_delta_x = 0.0
        self.accum_delta_y = 0.0
        self.accum_delta_facing = 0.0
        
    def _quantize_cardinal(self, theta):
        # nearest multiple of pi/2, normalized to [-pi, pi)
        k = int(np.round(theta / (np.pi/2.0)))
        return clip_rad_180(k * (np.pi/2.0))
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
            experience['place_kind'] = getattr(exp, 'place_kind', None)
            experience['room_color'] = getattr(exp, 'room_color', None)
            experience['grid_xy']    = getattr(exp, 'grid_xy', None)

            #experience['ob_info'] = exp.view_cell.template_info
            if wt_links == True:
                experience['links'] = exp.links

            map_experiences.append(experience)
        return map_experiences
    def _reanchor_to_current(self):
        """We just committed to self.current_exp as the anchor → zero deltas."""
        self.accum_delta_x = 0.0
        self.accum_delta_y = 0.0
        self.accum_delta_facing = 0.0
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
            experience['place_kind'] = getattr(exp, 'place_kind', None)
            experience['room_color'] = getattr(exp, 'room_color', None)
            experience['grid_xy']    = getattr(exp, 'grid_xy', None)
            #experience['ob_info'] = exp.view_cell.template_info
            #print('exp ' + str(self.current_exp.id) +' map position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) +' facing: ' +str(self.current_exp.facing_rad))
        return experience

    def get_delta_exp(self, x,y, delta_x, delta_y):
        ''' return euclidian distance between two points'''
        if self.current_exp is None:
            delta_exp = np.inf
        else:
            delta_exp = euclidian_distance([x,y], [delta_x, delta_y]) 
            
        
        return delta_exp   

    def accumulated_delta_location(self, vtrans, vrot):
        # Update delta heading first
        new_delta_facing = clip_rad_180(self.accum_delta_facing + vrot)

        # Heading to move along in MAP/world coordinates
        if self.current_exp is not None:
            theta_world = clip_rad_180(self.current_exp.facing_rad + new_delta_facing)
        else:
            theta_world = new_delta_facing  # no anchor yet

        new_delta_x = self.accum_delta_x + vtrans * np.cos(theta_world)
        new_delta_y = self.accum_delta_y + vtrans * np.sin(theta_world)

        return new_delta_facing, new_delta_x, new_delta_y

    def delta_exp_above_thresold(self, delta_exp:float)->bool:
        print('delta exp and delta threshold', delta_exp, self.DELTA_EXP_THRESHOLD)
        return delta_exp > self.DELTA_EXP_THRESHOLD
    
    def delta_pc_above_thresold(self, delta_pc):
        print('delta exp and delta threshold', delta_pc, self.DELTA_PC_THRESHOLD)
        return delta_pc > self.DELTA_PC_THRESHOLD

    def get_global_position(self):
        if self.current_exp is None:
            return (
                self.accum_delta_x,
                self.accum_delta_y,
                clip_rad_180(self.accum_delta_facing),
            )
        return (
            self.current_exp.x_m + self.accum_delta_x,
            self.current_exp.y_m + self.accum_delta_y,
            clip_rad_180(self.current_exp.facing_rad + self.accum_delta_facing),
        )
        
    
    def get_exp_global_position(self, exp:object=-1)->list:
        if isinstance(exp, int):
            if self.current_exp is not None:
                return [self.current_exp.x_m, self.current_exp.y_m, self.current_exp.facing_rad] 
            else:
                raise "get_exp_global_position can't accept element" + str(exp) +'of type' + str(type(exp))
        elif exp is None:
            raise "get_exp_global_position can't accept element" + str(exp) +'of type' + str(type(exp))
        return [exp.x_m, exp.y_m, exp.facing_rad] 
    # ────────────────────────────── MiniGrid tagging helpers ──────────────────────
    def _env_unwrapped(self, env):
        # Some wrappers hide grid; unwrapped always has .grid and .agent_pos
        return getattr(env, "unwrapped", env)

    def _get_agent_floor_cell(self, env):
        """
        Return (cell, (gx,gy)) where cell is the WorldObj under the agent,
        or None if out of bounds.
        """
        e = self._env_unwrapped(env)
        gx, gy = map(int, map(float, e.agent_pos))   # ensure ints
        if gx < 0 or gy < 0 or gx >= e.width or gy >= e.height:
            return None, (gx, gy)
        return e.grid.get(gx, gy), (gx, gy)

       # ------------------ Place/Color classification helpers ------------------

    def _env_unwrapped(self, env):
        return getattr(env, "unwrapped", env)

    def _neighbors4(self, x, y):
        return ((x+1,y), (x-1,y), (x,y+1), (x,y-1))

    def _is_floor(self, obj):
        # MiniGrid WorldObj: .type ∈ {'floor', 'wall', 'door', ...}
        return getattr(obj, "type", None) == "floor"

    def _is_corridor_color_name(self, color_name: str | None) -> bool:
        # In many ADRooms/Minigrid variants corridors are 'black', 'grey', or 'gray'
        if not color_name:
            return False
        c = color_name.lower()
        return c in ("black", "grey", "gray")

    def _flood_fill_floor(self, env, start_xy, max_tiles=400):
        """BFS over contiguous floor tiles; returns area, color histogram, and bounds."""
        from collections import deque, Counter
        e = self._env_unwrapped(env)
        W, H = e.width, e.height
        sx, sy = start_xy
        if not (0 <= sx < W and 0 <= sy < H):
            return {"area": 0, "colors": Counter(), "bounds": (sx, sx, sy, sy)}

        seen = set()
        q = deque([(sx, sy)])
        colors = Counter()
        xmin = xmax = sx
        ymin = ymax = sy

        while q and len(seen) < max_tiles:
            x, y = q.popleft()
            if (x, y) in seen:
                continue
            seen.add((x, y))
            cell = e.grid.get(x, y)
            if not self._is_floor(cell):
                continue

            colors[getattr(cell, "color", None)] += 1
            if x < xmin: xmin = x
            if x > xmax: xmax = x
            if y < ymin: ymin = y
            if y > ymax: ymax = y

            for nx, ny in self._neighbors4(x, y):
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen:
                    ncell = e.grid.get(nx, ny)
                    if self._is_floor(ncell):
                        q.append((nx, ny))

        return {"area": len(seen), "colors": colors, "bounds": (xmin, xmax, ymin, ymax)}

    def _classify_place_from_xy(self, env, gx, gy, *, area_room_threshold=None):
        """Classify (ROOM/CORRIDOR/UNKNOWN, color) at a specific grid cell."""
        e = self._env_unwrapped(env)
        if not (0 <= gx < e.width and 0 <= gy < e.height):
            return "UNKNOWN", None, (gx, gy)

        cell = e.grid.get(gx, gy)
        if not self._is_floor(cell):
            return "UNKNOWN", None, (gx, gy)

        color = getattr(cell, "color", None)
        # If it is an explicitly colored floor (not corridor palette), it's a ROOM
        if color and not self._is_corridor_color_name(color):
            return "ROOM", color, (gx, gy)

        # Otherwise use region size and dominant non-corridor color
        region = self._flood_fill_floor(env, (gx, gy))
        thr = self.area_room_threshold if area_room_threshold is None else area_room_threshold
        kind = "ROOM" if region["area"] >= thr else "CORRIDOR"

        # For ROOMs on neutral/black tiles, infer dominant non-corridor color in region
        dom = None
        if kind == "ROOM":
            # Filter out corridor palette; keep any true colors including 'purple'
            candidates = [(c, n) for c, n in region["colors"].items()
                          if c and not self._is_corridor_color_name(c)]
            if candidates:
                dom = max(candidates, key=lambda t: t[1])[0]
        return kind, dom, (gx, gy)

    def _classify_place_from_xy(self, env, gx, gy, *, area_room_threshold=None):
        """
        Classify an arbitrary (gx,gy). Uses metadata first, then door/corridor,
        then a robust bounds-based room fallback, then floor color as last resort.
        """
        try:
            e = self._env_unwrapped(env)
        except Exception:
            e = getattr(env, "unwrapped", env)

        gx = int(gx); gy = int(gy)

        # 1) Exact mapping via xy_to_room
        rid = None
        if hasattr(e, "get_room_id_at"):
            rid = e.get_room_id_at(gx, gy)
        elif hasattr(e, "xy_to_room"):
            rid = e.xy_to_room.get((gx, gy))
        if rid is not None:
            color = None
            if hasattr(e, "get_room_color_by_room"):
                color = e.get_room_color_by_room(rid)
            elif hasattr(e, "room_meta"):
                color = (e.room_meta.get(rid) or {}).get("color")
            return "ROOM", color, rid

        # 2) Door?
        is_door, door_color = False, None
        try:
            cell = e.grid.get(gx, gy) if (0 <= gx < e.width and 0 <= gy < e.height) else None
        except Exception:
            cell = None
        if cell is not None and getattr(cell, "type", None) == "door":
            is_door, door_color = True, getattr(cell, "color", None)
        if not is_door and hasattr(e, "door_xy"):
            is_door = (gx, gy) in set(e.door_xy)
        if is_door:
            return "DOOR", door_color, (gx, gy)

        # 3) Corridor?
        if hasattr(e, "corridor_xy") and (gx, gy) in e.corridor_xy:
            return "CORRIDOR", None, (gx, gy)

        # 4) Robust bounds-based room fallback (fills any mapping holes)
        if hasattr(e, "room_meta"):
            for rid2, meta in e.room_meta.items():
                b = meta.get("bounds")
                if not b or len(b) != 4:
                    continue
                x1, y1, x2, y2 = map(int, b)
                # inclusive bounds
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    return "ROOM", meta.get("color", None), rid2

        # 5) Last resort: tile-inspection (may classify black floor as ROOM — but only if we
        #    failed to identify corridor, which should be rare now that we register full stripes)
        if cell is not None and getattr(cell, "type", None) == "floor":
            return "ROOM", getattr(cell, "color", None), (gx, gy)

        return "UNKNOWN", None, (gx, gy)
    
    def _classify_place_at_agent(self, env, *, area_room_threshold=None):
        """
        Returns (place_kind, color, grid_xy)
        - place_kind ∈ {"ROOM","CORRIDOR","DOOR","UNKNOWN"}
        - color: room color (or door color) if known
        - grid_xy: (col,row) for ROOM (preferred) else (gx,gy)
        """
        try:
            e = self._env_unwrapped(env)
        except Exception:
            e = getattr(env, "unwrapped", env)

        try:
            gx, gy = map(int, map(float, getattr(e, "agent_pos", (None, None))))
        except Exception:
            gx = gy = None
        if gx is None or gy is None:
            return "UNKNOWN", None, None

        return self._classify_place_from_xy(env, gx, gy, area_room_threshold=area_room_threshold)


    def _annotate_place_from_env(self, exp, env):
        """
        Fill exp.place_kind, exp.room_color, exp.grid_xy using env's room/corridor map
        created at generation time. Falls back to tile-inspection if metadata is absent.
        """
        try:
            kind, color, gxy = self._classify_place_at_agent(env)
        except Exception:
            kind, color, gxy = "UNKNOWN", None, None

        exp.place_kind = kind
        exp.room_color = color
        exp.grid_xy    = gxy
        return exp
        
            

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
        prev_dx, prev_dy = self.accum_delta_x, self.accum_delta_y
        self.accum_delta_facing, self.accum_delta_x, self.accum_delta_y = self.accumulated_delta_location(vtrans, vrot)
        theta_snap(self, "post-integrate")
        theta_forward_check(self, vtrans, prev_dx, prev_dy, "post-integrate")
        
        #delta_prev_exp = self.get_delta_exp(0,0,self.accum_delta_x, self.accum_delta_y)
        #print('accumulated dist from prev exp: ' + str(self.accum_delta_x) + ' y: '+str(self.accum_delta_y) )
        
        #delta_exp_above_thresold = self.delta_exp_above_thresold(delta_prev_exp)
        adjust_map = False

        if self.current_exp != None:
        
            #approximate curent position
            current_GP = self.get_global_position()
            
            print('CHECK CURRENT GP X,Y,TH', current_GP )
            print("EXPS IN VIEW CELL", view_cell.exps, "CURRENT CELL",self.current_exp )
            print("VIEW CELL ID",view_cell.id, "CURRENT EXP VIEW CELL ID",self.current_exp.view_cell.id)
        
            delta_exps = []
        
            for e in self.exps:
                print("this are the experiences in self.exps",e,e.x_m,e.y_m,"id  ",e.view_cell.id)
                print(e.view_cell.id,e.view_cell.exps)
                print(e.view_cell.x_pc, e.view_cell.y_pc)
                for (i, e) in enumerate(e.view_cell.exps):
                    print('exp ' + str(i) +' view position x: ' + str(e.x_m) + ' y: '+str(e.y_m) )
                delta_exp = self.get_delta_exp(e.x_m,e.y_m, current_GP[0], current_GP[1])
                delta_exps.append(delta_exp)
            
            min_delta_GP_id = np.argmin(delta_exps)
            min_delta_GP_val = delta_exps[min_delta_GP_id]
            print('delta_exps',delta_exps,min_delta_GP_val)
            delta_pc = np.sqrt(
                min_delta(self.current_exp.x_pc, x_pc, self.DIM_XY)**2 +
                min_delta(self.current_exp.y_pc, y_pc, self.DIM_XY)**2 
                #+ min_delta(self.current_exp.th_pc, th_pc, self.DIM_TH)**2
                )
            print('current experience x: ',self.current_exp.x_m, ', y:',self.current_exp.y_m,' and th:', self.current_exp.facing_rad)
            print('current position x_pc: ',x_pc, ', y_pc:',y_pc,' and th_pc:', th_pc)
            print("current exp pose cells",self.current_exp.x_pc,self.current_exp.y_pc)
            print('delta_pc between exp and currenttty pc',delta_pc)

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

        print('accumulated dist in exp map', self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing,'is delta_exp_above_thresold?',min_delta_GP_val<self.DELTA_EXP_THRESHOLD, min_delta_GP_val,self.DELTA_EXP_THRESHOLD, 'closest exp dist', min_delta_GP_val, min_delta_GP_id)
        print('len exps =' ,len(view_cell.exps))
        # if current exp is None, just select first matching?
        if self.current_exp is None and Experience._ID == 0 :
            print('first experience created',local_pose)
            exp = self._create_exp(x_pc, y_pc, th_pc, view_cell, local_pose)

            self.current_exp = exp
            self.accum_delta_x = 0
            self.accum_delta_y = 0
            self.accum_delta_facing = 0
            #self.current_exp = view_cell.exps[0]
        #if we loaded a memory map, then we need to get experience matching view cell
        elif self.current_exp is None:
            self.current_exp = view_cell.exps[0] #NOTE: this works considering that 1 exp has 1 ob
            print('we are initialising position to',self.current_exp.id, 'extracted from , based on observation', view_cell.id)

            self.accum_delta_x = 0
            self.accum_delta_y = 0
            self.accum_delta_facing = self.current_exp.facing_rad
                   

        #We have a new view but it's close to a previous experience
        elif len(view_cell.exps) == 0 and min_delta_GP_val < (self.DELTA_EXP_THRESHOLD):
            print('too close from exp', min_delta_GP_id, ', dist between exp and here', min_delta_GP_val)
            print("current exp", self.current_exp)
            exp = self.get_exp(min_delta_GP_id)

            delta_exp_pc = np.sqrt(
                min_delta(self.current_exp.x_pc, exp.x_pc, self.DIM_XY)**2 +
                min_delta(self.current_exp.y_pc, exp.y_pc, self.DIM_XY)**2
                # + min_delta(self.current_exp.th_pc, exp.th_pc, self.DIM_TH)**2
            )
            print('checking motion between two considered experiences', delta_exp_pc)

            # Only perform loop‐closure if it's a different experience
            if min_delta_GP_id != self.current_exp.id:
                print('we are close looping with exp', min_delta_GP_id, 'discarding newly generated view_cell', view_cell.id)
                adjust_map = True
                close_loop_exp = self.exps[min_delta_GP_id]

                # see if the exp nearby already has a link to the current exp
                link_exists = any(l.target == close_loop_exp for l in self.current_exp.links)

                if not link_exists:
                    # link both ways
                    self.current_exp.link_to(
                        close_loop_exp,
                        self.accum_delta_x,
                        self.accum_delta_y,
                        close_loop_exp.facing_rad,
                        active_link=True
                    )
                    close_loop_exp.link_to(
                        self.current_exp,
                        -self.accum_delta_x,
                        -self.accum_delta_y,
                        self.current_exp.facing_rad,
                        active_link=False
                    )
                    theta_link_check(self.current_exp, close_loop_exp, self.current_exp.links[-1], "match->fwd")
                    theta_link_check(close_loop_exp, self.current_exp, close_loop_exp.links[-1], "match->back")


                # Remember current GP BEFORE changing anchor
                prev_gx, prev_gy, prev_gth = self.get_global_position()

                # commit to new anchor
                self.current_exp = close_loop_exp

                # set residuals so GP stays the same for x, y, *and* θ
                self.accum_delta_x = prev_gx  - close_loop_exp.x_m
                self.accum_delta_y = prev_gy  - close_loop_exp.y_m
                self.accum_delta_facing = clip_rad_180(prev_gth - close_loop_exp.facing_rad)

                print("Global Position:", self.get_global_position(),
                    self.current_exp.x_m, self.current_exp.y_m, self.current_exp.facing_rad)
                print('We keep current GP facing rad:', 
                    self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing)
            
        # if the vt is new AND the global pose x,y,th is far enough from any prev experience create a new experience
        elif len(view_cell.exps) == 0:
            #if current location is far enough from prev one, else, view cells are considered as in conflict
            print('no exp in view, len =' ,len(view_cell.exps), 'closest exp dist', min_delta_GP_val )
            exp = self._create_exp(x_pc, y_pc, th_pc, view_cell,local_pose)

                            
            self.current_exp = exp
            self.accum_delta_x = 0
            self.accum_delta_y = 0
            self.accum_delta_facing = 0
            
        

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
                print('exp ' + str(i) +' view position x: ' + str(e.x_m) + ' y: '+str(e.y_m) )
                
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
                if min_delta_GP_val < self.DELTA_EXP_THRESHOLD :
                    print('the delta between exp' , view_cell.exps[min_delta_GP_id].id ,' and ' ,self.current_exp.id, ' allow for a close-loop')
                    matched_exp = view_cell.exps[min_delta_GP_id]

                    # see if the prev exp already has a link to the current exp
                    link_exists = False
                    for linked_exp in [l.target for l in self.current_exp.links]:
                        if linked_exp == matched_exp:
                            link_exists = True
                            break

                    if not link_exists:
                        print(current_GP)
                        print("we are linking things",matched_exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing)
                        self.current_exp.link_to(
                            matched_exp,
                            self.accum_delta_x,
                            self.accum_delta_y,
                            matched_exp.facing_rad,        # <— target abs θ
                            active_link=True
                        )
                        matched_exp.link_to(
                            self.current_exp,
                            -self.accum_delta_x,           # <— negate
                            -self.accum_delta_y,
                            self.current_exp.facing_rad,   # <— target abs θ
                            active_link=False
                        )
                        # then snap and zero deltas
                        # Remember current GP BEFORE changing anchor
                        prev_gx, prev_gy, prev_gth = self.get_global_position()

                        # Commit to the new anchor
                        self.current_exp = matched_exp

                        # Re-carry residuals so GP stays EXACTLY the same
                        self.accum_delta_x      = prev_gx  - matched_exp.x_m
                        self.accum_delta_y      = prev_gy  - matched_exp.y_m
                        self.accum_delta_facing = clip_rad_180(prev_gth - matched_exp.facing_rad)
                if matched_exp is None:
                    if min_delta_GP_val > self.DELTA_EXP_THRESHOLD:
                        nearest_exp   = None
                        nearest_dist  = float("inf")
                        for e in self.exps:
                            d = self.get_delta_exp(e.x_m, e.y_m, current_GP[0], current_GP[1])
                            if d < nearest_dist:
                                nearest_dist, nearest_exp = d, e

                        if nearest_dist < self.DELTA_EXP_THRESHOLD:
                            print(f"[EM] SNAP to exp {nearest_exp.id}  (d={nearest_dist:.2f})")

                            # --- (A) Remember the old anchor and current GP before re-anchoring
                            prev_gx, prev_gy, prev_gth = self.get_global_position()

                            # --- (B) Optionally add a link old_exp -> matched_exp if it doesn't exist
                            
                            # before creating any link, right after computing nearest_exp / matched_exp
                            old_exp = self.current_exp
                            matched_exp = nearest_exp

                            if matched_exp is old_exp:
                                # We are already anchored on this node → no new link, no re-anchor churn.
                                # Optionally: keep residuals as-is, or zero them if you want strict snap.
                                # Here I keep residuals (continuous GP policy).
                                print(f"[EM] SNAP resolved to current exp {matched_exp.id} → no link/no-op")
                                # If you had planned to re-anchor math here, skip it and just return to the flow.
                            else:
                                # link-exists check (forward direction)
                                link_exists = any(l.target is matched_exp for l in old_exp.links)

                                # amount of *translational* odometry we just accumulated
                                d_trans = (self.accum_delta_x**2 + self.accum_delta_y**2) ** 0.5

                                # Gate: only add link if (1) not already present AND (2) we actually moved a bit
                                # You can tune eps; 1e-6 is numerically safe, 1e-3..1e-2 is pragmatic in grid envs
                                if (matched_exp is not old_exp) and (not link_exists) and (d_trans > 1e-6):
                                    # Forward link: old_exp -> matched_exp uses the ODOMETRY deltas
                                    old_exp.link_to(
                                        matched_exp,
                                        self.accum_delta_x,
                                        self.accum_delta_y,
                                        matched_exp.facing_rad,   # target absolute θ (see link_to implementation)
                                        active_link=True
                                    )

                                    # Backward link: matched_exp -> old_exp with negated deltas
                                    matched_exp.link_to(
                                        old_exp,
                                        -self.accum_delta_x,
                                        -self.accum_delta_y,
                                        old_exp.facing_rad,       # target absolute θ
                                        active_link=False
                                    )

                                # (Optional) tiny debug to verify internal consistency of the just-created links
                                # theta_link_check(old_exp, matched_exp, old_exp.links[-1], "snap->fwd")
                                # theta_link_check(matched_exp, old_exp, matched_exp.links[-1], "snap->back")

                                # --- (C) Re-anchor using your *continuous* policy so GP does not jump
                                self.current_exp = matched_exp
                                self.accum_delta_x       = prev_gx  - matched_exp.x_m
                                self.accum_delta_y       = prev_gy  - matched_exp.y_m
                                self.accum_delta_facing  = clip_rad_180(prev_gth - matched_exp.facing_rad)

                        else:
                            print("Creating new experience because no exp within metric gate")    
                            matched_exp = self._create_exp(x_pc, y_pc, th_pc,
                                                        view_cell_copy, local_pose)
                            self.current_exp   = matched_exp
                            self.accum_delta_x = self.accum_delta_y = 0
                            self.accum_delta_facing = 0.0 
                            print("warning, no matched experience although matching view cell ", min_delta_GP_val, "did we enter though?",nearest_dist > self.DELTA_EXP_THRESHOLD)
                    '''#View_cell_copy contains the same ob as view_cell but without experience attached and with a new ID
                    #TODO: add view_cell_copy instead of view_cell
                    matched_exp = self._create_exp(
                        x_pc, y_pc, th_pc, view_cell_copy,local_pose)
                    self.current_exp = matched_exp
                    self.accum_delta_x = 0
                    self.accum_delta_y = 0'''

                #self.current_exp = matched_exp
                #self.accum_delta_x = 0
                #self.accum_delta_y = 0
    
                #commented line below as we don't narrow close loop around position but place, this means no strict position correction in odom
                #self.accum_delta_facing = self.current_exp.facing_rad
        # print('current experience id and matching view',self.current_exp.id, view_cell.id)
        # print('exp ' + str(self.current_exp.id) +' map position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) +' facing: ' +str(self.current_exp.facing_rad))
        
        #If nothing particular and we are close from a previous experience, we link them if not already
        elif min_delta_GP_val < (0.3*self.DELTA_EXP_THRESHOLD) and close_exp_link_exists != True : 
            # if the closest exp has no link to the current exp
            print(current_GP)
            print('linking current exp ', self.current_exp.id, ' to closest exp ', min_delta_GP_id )
            min_delta_GP_exp = self.get_exp(min_delta_GP_id)

            delta_x = min_delta_GP_exp.x_m - self.current_exp.x_m
            delta_y = min_delta_GP_exp.y_m - self.current_exp.y_m

            self.current_exp.link_to(
                min_delta_GP_exp, delta_x, delta_y, min_delta_GP_exp.facing_rad, active_link=True
            )
            min_delta_GP_exp.link_to(
                self.current_exp, -delta_x, -delta_y, self.current_exp.facing_rad, active_link=False
            )

            

        #If we replaced the view cell in same exp, we need to update the global position to match init_local_position
        if self.current_exp.view_cell.to_update:
            print('Updating global location of the exp ', self.current_exp.id , 'to match the new init_local_position of view cell:',local_pose )

            print('OLD current exp position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) , ' facing: '+str(self.current_exp.facing_rad) \
                      , ' local pose: '+str(self.current_exp.init_local_position), 'view cell id', self.current_exp.view_cell.id ) 
            self.update_exp_wt_view_cell(self.current_exp, x_pc, y_pc, th_pc, new_view_cell= None, local_pose= local_pose)
            self.accum_delta_x=0
            self.accum_delta_y=0
            #self.accum_delta_facing = self.current_exp.facing_rad
            print('NEW current exp position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) , ' facing: '+str(self.current_exp.facing_rad) \
                , ' local pose: '+str(self.current_exp.init_local_position), 'view cell id', self.current_exp.view_cell.id) 
            self.current_exp.view_cell.to_update = False
            for link in self.current_exp.links:
                e1 = link.target
                print(current_GP)
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
        print("\n=== LINK DUMP (after step) ===")
        for e in self.exps:
            for l in e.links:
                print(f"{e.id:3}  → {l.target.id:3} | d={l.d:5.2f}  "
                    f"hdg={l.heading_rad:+5.2f}  fac={l.facing_rad:+5.2f} "
                    f"{'ACTIVE' if l.active_link else 'inv'}")
        print("=== END LINK DUMP ===\n")

        print('adjust map?', self.constant_adjust, adjust, adjust_map)
        
        do_adjust = (self.constant_adjust or (adjust and adjust_map))
        if not do_adjust:
            return
        if NO_DRIFT_MODE or not do_adjust:
            # If you want GP continuity across re-anchoring (what you already do),
            # keep the residuals as set above (prev_gx/gy/gth logic you already run).
            # If you prefer hard snap in a perfect world, uncomment the next 3 lines:
            # self.accum_delta_x = 0.0
            # self.accum_delta_y = 0.0
            # self.accum_delta_facing = 0.0

            theta_gp_check(self, "after-step")
            return
        prev_gx, prev_gy, prev_gth = self.get_global_position()
        self.relax_graph(fixed_exp=self.current_exp, loops=self.LOOPS)
        ax, ay, ath = self.current_exp.x_m, self.current_exp.y_m, self.current_exp.facing_rad
        self.accum_delta_x      = prev_gx  - ax
        self.accum_delta_y      = prev_gy  - ay
        self.accum_delta_facing = clip_rad_180(prev_gth - ath)
        for e0 in self.exps:           
            print('aFTER CORRECTIONS  exp ' + str(e0.id) +' map position x: ' + str(e0.x_m) + ' y: '+str(e0.y_m) +' facing: ' +str(e0.facing_rad))
          

        #print('AFTER CORRECTION  exp ' + str(self.current_exp.id) +' map position x: ' + str(self.current_exp.x_m) + ' y: '+str(self.current_exp.y_m) +' facing: ' +str(self.current_exp.facing_rad))
        # print( )
        # print('____')
        theta_gp_check(self, "after-step")
        return