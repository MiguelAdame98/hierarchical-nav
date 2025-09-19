import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time
import torch
from scipy import stats

class TrueHierarchicalHMMWithBOCPD:
    """
    True Hierarchical HMM where hidden states are (mode, submode) pairs,
    enhanced with Bayesian Online Changepoint Detection (BOCPD).
    This creates natural hierarchical structure with proper transition dynamics
    and adaptive changepoint detection.
    """
    def __init__(self, use_submodes: bool = False, use_bocpd: bool = True):
        """
        Simplified 4-mode configuration (EXPLORE, NAVIGATE, RECOVER, TASK_SOLVING).
        By default submodes are disabled to make the model tractable and debuggable.
        Set `use_submodes=True` if you want the original hierarchical (mode, submode) space.
        """
        self.use_submodes = use_submodes
        self.use_bocpd = use_bocpd

        self.current_mode='EXPLORE'
        # ---- Minimal 4-mode state space (call in __init__) -----------------
        self.states = ['EXPLORE', 'NAVIGATE', 'RECOVER', 'TASK_SOLVING']  # <- no submodes
        self.n_states = len(self.states)
        self.state_index = {m: i for i, m in enumerate(self.states)}
        self.state_beliefs = np.ones(self.n_states, dtype=float) / self.n_states

        # Optional gates + smoothing defaults (safe if previously absent)
        self.task_enter_gate = 1.0     # multiplier into TASK_SOLVING
        self.task_exit_gate  = 1.0     # multiplier out of TASK_SOLVING
        self.task_gate_fn    = None    # optional callable(self, evidence, tp_smooth) -> (enter, exit)
        self.task_progress_smooth = 0.0

        # Stickiness tuning knobs (used below)
        self.stickiness_decay_non_task = 0.40
        self.stickiness_task_bonus     = 0.03

        # Build your initial transition
        self.transition_matrix = self._build_transition_matrix()
        self.emission_params = self._build_emission_params()

        # === Beliefs ==========================================================
        self.state_beliefs = np.ones(self.n_states, dtype=float) / self.n_states
        self.state_history = deque(maxlen=500)

        # === BOCPD ============================================================
        self.max_run_length = 50
        self.run_length_dist = np.zeros(self.max_run_length + 1, dtype=float)
        self.run_length_dist[0] = 1.0
        self.hazard_rate = 1.0 / 10.0
        self.changepoint_threshold = 0.1

        # Evidence buffers
        self.evidence_buffer = deque(maxlen=100)
        self.evidence_history = deque(maxlen=self.max_run_length)
        self.mode_history = deque(maxlen=100)

        # Counters / diagnostics
        self.step_counter = 0
        self.stagnation_counter = 0
        self.exploration_attempts = 0
        self.navigation_attempts = 0
        self.recovery_attempts = 0
        self._seen_node_ids: set[str] = set()   # places we've seen historically
        self._last_node_id: str | None = None   # for most-recent switch detection
        # If no node IDs are in the replay buffer, we quantize (x,y) to a grid:
        self._node_quantize_m: float = 0.75     # grid size (meters) for fallback place IDs

        self.debug_evidence=True
        self.debug_emissions=True

        self.prev_evidence = None
        self.params = {
            'loop_threshold': 20,
            'stagnation_time_threshold': 30.0,
            'stagnation_step_threshold': 50,
            'max_exploration_cycles': 3,
            'max_navigation_cycles': 2,
            'max_recovery_cycles': 2
        }

    def _build_transition_matrix(self) -> np.ndarray:
        """
        4x4 transition matrix with high diagonals (stickiness) and gate-controlled
        off-diagonals. Rows are normalized; safe fallback to uniform if needed.
        """
        n = self.n_states
        T = np.zeros((n, n), dtype=float)
        for i, fm in enumerate(self.states):
            row = np.zeros(n, dtype=float)
            # stay
            row[i] = self._mode_stickiness(fm)
            # move to others
            for j, tm in enumerate(self.states):
                if j == i: 
                    continue
                row[j] = self._get_inter_mode_transition(fm, tm)

            # normalize safely
            s = float(row.sum())
            if not np.isfinite(s) or s <= 0.0:
                row[:] = 1.0 / n
            else:
                row /= s
            T[i, :] = row

        # final sanity
        if not np.all(np.isfinite(T)):
            print("[WARN] Non-finite entries in T; forcing uniform.")
            T[:] = 1.0 / n
        return T    
    def _logpdf_normal_np(self, x: float, mu: float, sigma: float) -> float:
        """NumPy-only normal logpdf (fallback when SciPy is unavailable)."""
        sigma = max(float(sigma), 1e-6)
        z = (float(x) - float(mu)) / sigma
        return float(-0.5*np.log(2.0*np.pi) - np.log(sigma) - 0.5*z*z)

    def _gauss_logpdf(self, x: float, mean: float, std: float) -> float:
        """Try SciPy; otherwise use the NumPy fallback."""
        try:
            from scipy.stats import norm
            return float(norm.logpdf(x, loc=mean, scale=max(std, 1e-6)))
        except Exception:
            return self._logpdf_normal_np(x, mean, std)
    def _build_hierarchical_transition_matrix(self) -> np.ndarray:
        """
        Build transition matrix with hierarchical structure. When submodes are disabled
        this reduces to a 4x4 mode-level HMM with strong diagonals ("stickiness").
        """
        T = np.zeros((self.n_states, self.n_states), dtype=float)

        for i, (from_mode, from_submode) in enumerate(self.hierarchical_states):
            row = np.zeros(self.n_states, dtype=float)

            for j, (to_mode, to_submode) in enumerate(self.hierarchical_states):
                if from_mode == to_mode:
                    if from_submode == to_submode:
                        row[j] += self._get_submode_persistence(from_mode, from_submode)
                    else:
                        row[j] += self._get_intra_mode_transition(from_mode, from_submode, to_submode)
                else:
                    row[j] += self._get_inter_mode_transition(from_mode, from_submode, to_mode, to_submode)

            # Ensure positivity and normalise
            row = np.maximum(row, 1e-12)
            row_sum = row.sum()
            if row_sum <= 0.0:
                # Fallback to uniform
                row[:] = 1.0 / self.n_states
            else:
                row /= row_sum
            T[i, :] = row

        return T
    
    
    def _get_intra_mode_transition(self, mode: str, from_sub: str = None, to_sub: str = None) -> float:
        """
        Submode transitions are disabled (no submodes anymore).
        Keep signature for compatibility; always returns 0.
        """
        return 0.0
    
    def _get_inter_mode_transition(self, from_mode: str, to_mode: str) -> float:
        """
        Unnormalized inter-mode transition weight with TASK_SOLVING gates applied.
        """
        base = {
            'EXPLORE':      {'NAVIGATE': 0.45, 'RECOVER': 0.09, 'TASK_SOLVING': 0.01},
            'NAVIGATE':     {'EXPLORE': 0.79, 'RECOVER': 0.08, 'TASK_SOLVING': 0.01},
            'RECOVER':      {'EXPLORE': 0.80, 'NAVIGATE': 0.20, 'TASK_SOLVING': 0.01},
            'TASK_SOLVING': {'EXPLORE': 0.05, 'NAVIGATE': 0.01, 'RECOVER': 0.09},
        }
        w = float(base.get(from_mode, {}).get(to_mode, 0.0))
        if w <= 0.0:
            return 0.0

        if to_mode == 'TASK_SOLVING' and from_mode != 'TASK_SOLVING':
            w *= float(max(self.task_enter_gate, 1e-6))
        if from_mode == 'TASK_SOLVING' and to_mode != 'TASK_SOLVING':
            w *= float(max(self.task_exit_gate, 1e-6))

        return float(max(w, 0.0))
    
    def set_task_solving_gate(self, enter: float = None, exit: float = None, fn=None) -> None:
        """
        Configure how strongly the model moves into/out of TASK_SOLVING.
        - enter: scalar multiplier for transitions INTO TASK_SOLVING  (default 1.0)
        - exit : scalar multiplier for transitions OUT OF TASK_SOLVING (default 1.0)
        - fn   : optional callable (self, evidence, tp_smooth) -> (enter, exit)
                If provided, it overrides fixed gates and will be called each update().

        Example:
            hhmm.set_task_solving_gate(enter=0.02, exit=0.6)  # very reluctant to enter
            hhmm.set_task_solving_gate(fn=my_gate)            # use a custom schedule
        """
        if enter is not None:
            self.task_enter_gate = float(np.clip(enter, 1e-6, 1e6))
        if exit is not None:
            self.task_exit_gate  = float(np.clip(exit,  1e-6, 1e6))
        if fn is not None and callable(fn):
            self.task_gate_fn = fn
    def _build_emission_params(self) -> Dict[str, Dict]:
        """
        Mode-keyed emission parameters (no submodes).
        Means/σ assume evidence features are roughly in [0,1].
        Tune from logs once the pipeline runs.
        """
        params = {
            'EXPLORE': {
                'lost_prob': 0.20,
                'loop_prob': 0.20,
                'stagnation':               {'mean': 0.25, 'std': 0.15},
                'info_gain':                {'mean': 0.45, 'std': 0.22},
                'exploration_productivity': {'mean': 0.40, 'std': 0.25},
                'known_switch_prob': 0.15,                        # EXPLORE expects NOT switching into known places
                'known_revisit_rate': {'mean': 0.05, 'std': 0.12} # revisit rate should be low in exploration

            },
            'NAVIGATE': {
                'lost_prob': 0.15,
                'loop_prob': 0.12,
                'stagnation':          {'mean': 0.25, 'std': 0.18},
                'navigation_progress': {'mean': 0.70, 'std': 0.25},
                'known_switch_prob': 0.70,                        # NAVIGATE often switches between known nodes
                'known_revisit_rate': {'mean': 0.60, 'std': 0.25} # higher revisit ratio during nav is fine

            },
            'RECOVER': {
                'lost_prob': 0.85,
                'loop_prob': 0.60,
                'stagnation':             {'mean': 0.65, 'std': 0.18},
                'recovery_effectiveness': {'mean': 0.60, 'std': 0.18},
            },
            'TASK_SOLVING': {
                'lost_prob': 0.12,
                'loop_prob': 0.12,
                'stagnation':    {'mean': 0.12, 'std': 0.08},
                'task_progress': {'mean': 0.65, 'std': 0.18},   # swap to 'plan_progress' if that’s your signal
            },
        }
        return params

    
    def compute_emission_likelihood(self, evidence: Dict, state: str) -> float:
        """
        Emission likelihood p(evidence | state=mode).
        - 'state' is a MODE string: one of {'EXPLORE','NAVIGATE','RECOVER','TASK_SOLVING'}.
        - Uses Bernoulli terms for booleans (agent_lost, loop_detected) when configured.
        - Uses Gaussian log-likelihoods (SciPy) for continuous features when both
        the evidence key and the param key exist.
        - Missing features contribute neutrally (no penalty/bonus).
        - Returns a probability in (0, 1], floored to avoid exact zeros.
        """

        # Guard: must have parameters for this mode
        if not hasattr(self, 'emission_params') or state not in self.emission_params:
            return 1e-10

        params = self.emission_params[state]
        log_ll = 0.0
        contrib_dbg = []  # optional debug contributions

        # --- Universal Bernoulli features (if present) --------------------
        if 'agent_lost' in evidence and 'lost_prob' in params:
            p = float(np.clip(params['lost_prob'], 1e-10, 1 - 1e-10))
            term = np.log(p if evidence['agent_lost'] else (1 - p))
            log_ll += term
            if getattr(self, 'debug_emissions', False):
                contrib_dbg.append(('agent_lost', evidence['agent_lost'], p, term))

        if 'loop_detected' in evidence and 'loop_prob' in params:
            p = float(np.clip(params['loop_prob'], 1e-10, 1 - 1e-10))
            term = np.log(p if evidence['loop_detected'] else (1 - p))
            log_ll += term
            if getattr(self, 'debug_emissions', False):
                contrib_dbg.append(('loop_detected', evidence['loop_detected'], p, term))

        # Optional: stagnation as Gaussian if provided
        if 'stagnation' in evidence and 'stagnation' in params:
            x   = float(evidence['stagnation'])
            mu  = float(params['stagnation']['mean'])
            sig = max(float(params['stagnation']['std']), 1e-3)
            term = stats.norm.logpdf(x, mu, sig)
            log_ll += term
            if getattr(self, 'debug_emissions', False):
                contrib_dbg.append(('stagnation', (x, mu, sig), None, term))

        if 'recent_known_switch' in evidence and 'known_switch_prob' in params:
            p = float(np.clip(params['known_switch_prob'], 1e-10, 1 - 1e-10))
            term = np.log(p if evidence['recent_known_switch'] else (1 - p))
            log_ll += term
            if getattr(self, 'debug_emissions', False):
                contrib_dbg.append(('recent_known_switch', evidence['recent_known_switch'], p, term))


        if 'known_revisit_rate' in evidence and 'known_revisit_rate' in params:
            x   = float(evidence['known_revisit_rate'])
            mu  = float(params['known_revisit_rate']['mean'])
            sig = max(float(params['known_revisit_rate']['std']), 1e-3)
            term = stats.norm.logpdf(x, mu, sig)
            log_ll += term
            if getattr(self, 'debug_emissions', False):
                contrib_dbg.append(('known_revisit_rate', (x, mu, sig), None, term))
        # --- Mode-specific continuous features ----------------------------
        if state == 'EXPLORE':
            if 'info_gain' in evidence and 'info_gain' in params:
                x   = float(evidence['info_gain'])
                mu  = float(params['info_gain']['mean'])
                sig = max(float(params['info_gain']['std']), 1e-3)
                term = stats.norm.logpdf(x, mu, sig)
                log_ll += term
                if getattr(self, 'debug_emissions', False):
                    contrib_dbg.append(('info_gain', (x, mu, sig), None, term))
            if 'exploration_productivity' in evidence and 'exploration_productivity' in params:
                x   = float(evidence['exploration_productivity'])
                mu  = float(params['exploration_productivity']['mean'])
                sig = max(float(params['exploration_productivity']['std']), 1e-3)
                term = stats.norm.logpdf(x, mu, sig)
                log_ll += term
                if getattr(self, 'debug_emissions', False):
                    contrib_dbg.append(('exploration_productivity', (x, mu, sig), None, term))

        elif state == 'NAVIGATE':
            if 'navigation_progress' in evidence and 'navigation_progress' in params:
                x   = float(evidence['navigation_progress'])
                mu  = float(params['navigation_progress']['mean'])
                sig = max(float(params['navigation_progress']['std']), 1e-3)
                term = stats.norm.logpdf(x, mu, sig)
                log_ll += term
                if getattr(self, 'debug_emissions', False):
                    contrib_dbg.append(('navigation_progress', (x, mu, sig), None, term))

        elif state == 'RECOVER':
            if 'recovery_effectiveness' in evidence and 'recovery_effectiveness' in params:
                x   = float(evidence['recovery_effectiveness'])
                mu  = float(params['recovery_effectiveness']['mean'])
                sig = max(float(params['recovery_effectiveness']['std']), 1e-3)
                term = stats.norm.logpdf(x, mu, sig)
                log_ll += term
                if not evidence.get('agent_lost', False):
                    # multiply likelihood by a small factor (e.g., 1e-3)
                    penalty = float(getattr(self, 'recover_notlost_penalty', 1e-3))
                    log_ll += np.log(np.clip(penalty, 1e-12, 1.0))
                if getattr(self, 'debug_emissions', False):
                    contrib_dbg.append(('recovery_effectiveness', (x, mu, sig), None, term))

        elif state == 'TASK_SOLVING':
            # If your pipeline produces 'plan_progress' instead of 'task_progress',
            # mirror it into evidence earlier OR change the key here.
            key = 'task_progress' if 'task_progress' in params else None
            if key and key in evidence:
                x   = float(evidence[key])
                mu  = float(params[key]['mean'])
                sig = max(float(params[key]['std']), 1e-3)
                term = stats.norm.logpdf(x, mu, sig)
                log_ll += term
                if getattr(self, 'debug_emissions', False):
                    contrib_dbg.append((key, (x, mu, sig), None, term))

        # --- Finalize ------------------------------------------------------
        # Numerical safety (avoids underflow to exactly 0)
        log_ll = float(np.clip(log_ll, -700.0, 50.0))  # -700 ~ exp underflow boundary for float64
        lik = float(np.exp(log_ll))
        lik = max(lik, 1e-12)

        if getattr(self, 'debug_emissions', False):
            print(f"[EMIT] state={state}  logL={log_ll:.3f}  lik={lik:.3e}  contribs={contrib_dbg}")

        return lik
    def bind_env(self, env):
        """Call this once after you create the HMM (e.g., after gym.make)."""
        self._bound_env = env
        # optional reset state on new env:
        self._coverage_hist = None
        self.task_external_ready = False

    def _env_room_metrics(self, env):
        """
        Return (coverage in [0,1], complete flag) using the env’s own room bookkeeping.
        Works with aisle_door_rooms.* which expose:
        - get_visited_rooms_order(): [{'room': (col,row), ...}, ...]
        - rooms_in_row, rooms_in_col
        """
        e = getattr(env, "unwrapped", env)

        visited_ids = set()
        # Authoritative history (ordered)
        try:
            for rec in list(e.get_visited_rooms_order()):
                rid = rec.get("room")
                if isinstance(rid, (tuple, list)) and len(rid) == 2:
                    visited_ids.add((int(rid[0]), int(rid[1])))
        except Exception:
            pass

        # Optional fast path: fuse any private visited set, if present
        try:
            vset = getattr(e, "_visited_rooms_set", None)
            if isinstance(vset, set):
                visited_ids |= {tuple(v) if isinstance(v, (list, tuple)) else v for v in vset}
        except Exception:
            pass

        # Target total rooms from env metadata
        try:
            n_row = int(getattr(e, "rooms_in_row"))
            n_col = int(getattr(e, "rooms_in_col"))
            total = max(1, n_row * n_col)
        except Exception:
            # Last resort: avoid deadlock if metadata is missing
            total = max(1, len(visited_ids))

        visited = len(visited_ids)
        coverage = min(1.0, visited / float(total))
        complete = (visited >= total)
        return coverage, complete

    def _update_task_flag_from_env(self):
        """
        Decides when to raise `task_external_ready` using only env coverage.
        Primary rule: arm when rooms are complete.
        Optional fallback: arm on saturation (coverage stops increasing while revisits are high).
        """
        env = getattr(self, "_bound_env", None)
        if env is None:
            return None, False  # no env bound

        cov, complete = self._env_room_metrics(env)

        # Expose simple rolling history for a saturation fallback (optional)
        import collections
        if not hasattr(self, "_coverage_hist") or self._coverage_hist is None:
            self._coverage_hist = collections.deque(maxlen=int(getattr(self, "coverage_saturation_k", 20)))
        self._coverage_hist.append(float(cov))

        # Config knobs
        cov_arm_thresh   = float(getattr(self, "coverage_arm_threshold", 1.0))   # usually 1.0 (all rooms)
        sat_eps          = float(getattr(self, "coverage_saturation_eps", 0.002))# “no growth” band
        sat_rev_thresh   = float(getattr(self, "coverage_saturation_rev", 0)) # “we churn” threshold

        # Pull revisit intensity if available
        mean_rev = float(getattr(self, "_last_known_revisit_rate", 0.0))  # we'll set this below

        # Primary: exact completion (preferred)
        if complete or cov >= cov_arm_thresh:
            print("ARMED",complete,cov,cov_arm_thresh)
            self.task_external_ready = True
            return cov, True

        

        # Otherwise not armed
        return cov, False
    
    def _apply_post_nav_grace(self, evidence: dict, movement_metrics: dict | None = None, dbg: bool = False) -> None:
        """
        Arm and apply a 'post-navigation grace' window based on sustained low
        navigation_progress while the HMM believes we are in NAVIGATE.
        Mutates `evidence` in-place. Safe to call every step.
        """

        # ---- knobs (instance-overridable) ----
        nav_done_low_threshold = float(getattr(self, 'nav_done_low_threshold', 0.15))  # nav considered "low"
        nav_done_low_steps     = int(getattr(self,  'nav_done_low_steps',     2))      # sustain low K steps (while in NAV)
        post_nav_grace_steps   = int(getattr(self,  'post_nav_grace_steps',   15))     # grace length (applies in any mode)

        grace_kr_scale         = float(getattr(self, 'post_nav_grace_kr_scale', 0.0))  # damp revisit pressure
        grace_nav_cap          = float(getattr(self, 'post_nav_grace_nav_cap',  0.20)) # cap nav_progress
        grace_ig_floor         = float(getattr(self, 'post_nav_grace_info_gain',0.45)) # boost IG
        grace_stag_cap         = float(getattr(self, 'post_nav_grace_stag_cap', 0.45)) # reduce stagnation
        # --------------------------------------

        # Robust mode read: prefer evidence['mode'] if you pass it; else use attribute
        mode_now = (evidence.get('mode')
                    or getattr(self, 'current_mode', 'EXPLORE'))
        in_nav   = (str(mode_now).upper() == 'NAVIGATE')

        # grade is already clamped {0,1} in your runner
        nav_prog = float(evidence.get('navigation_progress', 0.0))

        # Stateful counters (persist across steps)
        if not hasattr(self, '_nav_low_age'):
            self._nav_low_age = 0
        if not hasattr(self, '_post_nav_grace_left'):
            self._post_nav_grace_left = 0
        if not hasattr(self, '_last_mode'):
            self._last_mode = mode_now

        # ========= ARMER (while in NAV only) =========
        if in_nav:
            if nav_prog <= nav_done_low_threshold:
                self._nav_low_age += 1
            else:
                if self._nav_low_age and dbg:
                    print(f"[HMM-GRACE] reset: nav_prog={nav_prog:.2f} > thr={nav_done_low_threshold:.2f} (age was {self._nav_low_age})")
                self._nav_low_age = 0

            if (self._post_nav_grace_left == 0) and (self._nav_low_age >= nav_done_low_steps):
                self._post_nav_grace_left = post_nav_grace_steps
                self._nav_low_age = 0  # optional: clear once armed
                if dbg:
                    print(f"[HMM-GRACE] ARMED (in NAV): ttl={self._post_nav_grace_left} "
                        f"(nav_prog≤{nav_done_low_threshold:.2f} for {nav_done_low_steps} steps)")
        else:
            # Outside NAV we do *not* increment or reset the detector.
            # But we *do* allow already-armed grace to continue counting down.
            pass

        # ========= OPTIONAL: instant arm on explicit finish flag =========
        # If you later decide to pass a boolean like evidence['nav_plan_finished'], this will arm immediately.
        if (self._post_nav_grace_left == 0) and evidence.get('nav_plan_finished', False):
            self._post_nav_grace_left = post_nav_grace_steps
            self._nav_low_age = 0
            if dbg:
                print(f"[HMM-GRACE] ARMED (finish flag): ttl={self._post_nav_grace_left}")

        # ========= APPLY GRACE (in any mode, once armed) =========
        if self._post_nav_grace_left > 0:
            self._post_nav_grace_left -= 1

            # capture pre-values for debug
            _kr0   = evidence.get('known_revisit_rate', None)
            _nav0  = evidence.get('navigation_progress', None)
            _ig0   = evidence.get('info_gain', None)
            _stag0 = evidence.get('stagnation', None)

            # 1) Dampen revisit pressure so EXPLORE isn’t crushed near the goal
            if 'known_revisit_rate' in evidence and _kr0 is not None:
                evidence['known_revisit_rate'] = float(_kr0) * grace_kr_scale

            # 2) Cap nav signal so NAV loses emission dominance during grace
            if 'navigation_progress' in evidence and _nav0 is not None:
                evidence['navigation_progress'] = min(float(_nav0), grace_nav_cap)

            # 3) Make EXPLORE attractive for a short burst
            if 'info_gain' in evidence and _ig0 is not None:
                evidence['info_gain'] = max(float(_ig0), grace_ig_floor)
            if 'stagnation' in evidence and _stag0 is not None:
                evidence['stagnation'] = min(float(_stag0), grace_stag_cap)

            # 4) Keep derived feature consistent
            try:
                expl_factor = float((movement_metrics or {}).get('exploration_factor', 0.5))
            except Exception:
                expl_factor = 0.5
            evidence['exploration_productivity'] = (
                float(evidence.get('info_gain', 0.0)) * (1.0 - float(evidence.get('stagnation', 0.0))) * expl_factor
            )

            if dbg:
                print(f"[HMM-GRACE] ACTIVE ttl={self._post_nav_grace_left} in_nav={in_nav} | "
                    f"KR {None if _kr0 is None else f'{float(_kr0):.2f}'}→{evidence.get('known_revisit_rate','∅')}  "
                    f"NAV {None if _nav0 is None else f'{float(_nav0):.2f}'}→{evidence.get('navigation_progress','∅')}  "
                    f"IG  {None if _ig0  is None else f'{float(_ig0):.2f}'}→{evidence.get('info_gain','∅')}  "
                    f"STG {None if _stag0 is None else f'{float(_stag0):.2f}'}→{evidence.get('stagnation','∅')}")
        else:
            # Only clear detector when grace has finished AND we're no longer in NAV.
            # (If we’re still in NAV we want to keep counting consecutive low steps.)
            if (not in_nav) and (self._nav_low_age != 0):
                if dbg:
                    print(f"[HMM-GRACE] detector cleared outside NAV (age was {self._nav_low_age})")
                self._nav_low_age = 0

        self._last_mode = mode_now


    def extract_evidence_from_replay_buffer(self, replay_buffer, external_info_gain=None, external_plan_progress=None):
        """
        Centralized evidence extraction that calculates all metrics once and reuses them.
        This replaces multiple separate calculations with a single comprehensive analysis.
        """
        dbg = bool(getattr(self, "debug_evidence", True))
        if dbg:
            print(f"[EVD] extract_evidence: len(buffer)={len(replay_buffer)}  "
                f"external_info_gain={external_info_gain}  external_plan_progress={external_plan_progress}")

        evidence = {}

        # ---------- WARM-UP SHORT PATH ----------
        if len(replay_buffer) < 10:
            warmup = {
                'agent_lost': False,
                'loop_detected': False,
                # keep movement/stagnation middling but not extreme
                'movement_score': 0.7,
                'stagnation': 0.4,                      # slightly below 0.5 so EXPLORE isn't penalized hard
                # let HMM compute internal info gain later; do not over-penalize EXPLORE
                'info_gain': 0.7,
                'exploration_productivity': 0.5,        # modest; depends on info_gain and (1 - stagnation)
                # DO NOT seed NAVIGATE/TASK with 0.5 at warm-up; make them low until we know better
                'navigation_progress': 0.1,
                'plan_progress': 0.1,
                'task_progress': 0.1,
                # RECOVER shouldn't look good at warm-up
                'recovery_effectiveness': 0.1,
                'new_node_created': False,
                'recent_known_switch': False,
                'known_revisit_rate': 0.0,
                '_warmup': True                         # <-- tag for update()
            }
            if dbg:
                print("[EVD] warm-up: using neutral defaults (no metrics computed).")
                print("[EVD] warm-up evidence:", warmup)
            return warmup

        # ---------- FULL PATH ----------
        # Convert replay buffer to list for easier manipulation
        buffer_list = list(replay_buffer)
        recent_entries = buffer_list[-10:]  # Last 10 entries for most calculations
        if dbg:
            print(f"[EVD] using last N={len(recent_entries)} entries for metrics.")

        # ===== CENTRALIZED METRIC CALCULATIONS =====

        # 1) Extract poses and positions
        poses = []
        positions = []
        def _coerce_xytheta(p):
            """Return (x,y,theta) as floats or None if not parseable."""
            try:
                # dict
                if isinstance(p, dict) and ('x' in p and 'y' in p):
                    return (float(p['x']), float(p['y']), float(p.get('theta', 0.0)))
                # list/tuple
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x = float(p[0]); y = float(p[1]); th = float(p[2]) if len(p) >= 3 else 0.0
                    return (x, y, th)
                # numpy
                if isinstance(p, np.ndarray) and p.size >= 2:
                    p = p.astype(float).copy()  # copy to break aliasing views
                    x = float(p[0]); y = float(p[1]); th = float(p[2]) if p.size >= 3 else 0.0
                    return (x, y, th)
                # torch
                if 'torch' in globals():
                    torch = globals()['torch']
                    if hasattr(torch, 'is_tensor') and torch.is_tensor(p) and p.numel() >= 2:
                        q = p.detach().cpu().float().clone().numpy()  # clone → new storage
                        x = float(q[0]); y = float(q[1]); th = float(q[2]) if q.size >= 3 else 0.0
                        return (x, y, th)
            except Exception:
                return None
            return None
        for entry in recent_entries:
            if 'real_pose' in entry:
                p = entry['real_pose']

                # Accept dict
                if isinstance(p, dict) and {'x', 'y'}.issubset(p):
                    poses.append((p['x'], p['y'], p.get('theta', 0.0)))
                    positions.append((float(p['x']), float(p['y'])))

                # Accept list / tuple
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    poses.append(p)
                    positions.append((float(p[0]), float(p[1])))

                # Accept numpy / tensor
                elif isinstance(p, np.ndarray) and p.size >= 2:
                    p = p.astype(float)
                    poses.append(p.tolist())
                    positions.append((p[0], p[1]))

                # Accept torch tensor (only if torch is available in caller)
                elif 'torch' in globals() and hasattr(globals()['torch'], 'is_tensor') and globals()['torch'].is_tensor(p) and p.numel() >= 2:
                    p = p.detach().cpu().float().tolist()
                    poses.append(p)
                    positions.append((p[0], p[1]))
        if dbg:
            sample_pos = positions[-1] if positions else None
            print(f"[EVD] extracted poses={len(poses)}  positions={len(positions)}  last_position={sample_pos}",poses,positions)

        # 2) Extract doubt counts (primary recovery/lost indicator)
        doubt_counts = []
        current_doubt_count = 0
        for entry in recent_entries:
            if 'place_doubt_step_count' in entry:
                doubt_count = entry['place_doubt_step_count']
                doubt_counts.append(doubt_count)
                current_doubt_count = doubt_count  # Keep updating to get latest
        if dbg:
            print(f"[EVD] doubt_counts(len)={len(doubt_counts)}  current_doubt={current_doubt_count}  "
                f"recent5={doubt_counts[-5:] if len(doubt_counts)>=5 else doubt_counts}")

        # 3) Calculate movement metrics
        movement_metrics = self._calculate_movement_metrics(poses, positions)
        if dbg:
            mm = movement_metrics
            print("[EVD] movement_metrics:",
                {k: round(float(mm[k]), 3) if isinstance(mm.get(k, None), (int, float, np.floating)) else mm.get(k)
                for k in mm.keys() if k in ('movement_score', 'exploration_factor', 'speed', 'turn_rate')})

        # 4) Calculate position analysis
        position_metrics = self._calculate_position_metrics(positions, recent_entries)
        if dbg:
            pm = position_metrics
            # Print a few salient fields if present
            to_show = {k: pm.get(k) for k in ('position_stagnation', 'unique_cells_last_k', 'loop_candidates', 'pose_variance')}
            print("[EVD] position_metrics:", to_show)

        # 5) Calculate doubt trend analysis
        doubt_metrics = self._calculate_doubt_metrics(doubt_counts)
        if dbg:
            dm = doubt_metrics
            to_show_dm = {k: dm.get(k) for k in ('current_doubt', 'trend', 'spikes', 'mean')}
            print("[EVD] doubt_metrics:", to_show_dm)

        # ===== BUILD EVIDENCE USING CENTRALIZED METRICS =====
        revisit = self._calculate_revisit_metrics(recent_entries, positions)
        evidence['recent_known_switch'] = revisit['recent_known_switch']
        evidence['known_revisit_rate']  = revisit['known_revisit_rate']
        if dbg:
            print(f"[EVD] revisit_metrics: recent_known_switch={evidence['recent_known_switch']}  "
                f"known_revisit_rate={evidence['known_revisit_rate']:.2f}")
        self._last_known_revisit_rate = float(evidence['known_revisit_rate'])
        # Universal evidence
        evidence['agent_lost'] = self._determine_agent_lost(
            current_doubt_count, movement_metrics, position_metrics
        )
        evidence['loop_detected'] = self._determine_loop_detected(
            position_metrics, buffer_list
        )
        evidence['movement_score'] = movement_metrics['movement_score']
        evidence['stagnation'] = 1.0 - movement_metrics['movement_score']

        if dbg:
            print(f"[EVD] universal: agent_lost={evidence['agent_lost']}  loop_detected={evidence['loop_detected']}  "
                f"movement_score={round(float(evidence['movement_score']),3)}  "
                f"stagnation={round(float(evidence['stagnation']),3)}")

        # Mode-specific evidence with external inputs (info_gain / exploration_productivity)
        if external_info_gain is not None:
            evidence['info_gain'] = float(external_info_gain)
            evidence['exploration_productivity'] = (
                evidence['info_gain'] * (1.0 - evidence['stagnation']) *
                movement_metrics['exploration_factor']
            )
            if dbg:
                print("[EVD] info_gain source: EXTERNAL  "
                    f"info_gain={round(float(evidence['info_gain']),3)}  "
                    f"exploration_factor={round(float(movement_metrics['exploration_factor']),3)}  "
                    f"exploration_productivity={round(float(evidence['exploration_productivity']),3)}")
        else:
            evidence['info_gain'] = movement_metrics['exploration_factor'] * 0.6
            evidence['exploration_productivity'] = evidence['info_gain'] * (1.0 - evidence['stagnation'])
            if dbg:
                print("[EVD] info_gain source: INTERNAL_MOVEMENT  "
                    f"exploration_factor={round(float(movement_metrics['exploration_factor']),3)}  "
                    f"info_gain={round(float(evidence['info_gain']),3)}  "
                    f"exploration_productivity={round(float(evidence['exploration_productivity']),3)}")

        # Plan / task progress (external vs internal estimate)
        if external_plan_progress is not None:
            progress_value = float(external_plan_progress)
            # down-weight external progress when we’re clearly still exploring
            alpha = 1.0
            if evidence.get('stagnation', 1.0) < 0.35 and movement_metrics.get('exploration_factor',0.0) > 0.5:
                alpha = 0.5  # halve the effect while exploration is productive
            evidence['navigation_progress'] = alpha * progress_value
            evidence['plan_progress']       = alpha * progress_value
            evidence['task_progress']       = alpha * progress_value
            if dbg:
                print(f"[EVD] progress source: EXTERNAL  value={round(progress_value,3)}")
        else:
            estimated_progress = self._estimate_progress_from_movement(
                movement_metrics, doubt_metrics, position_metrics
            )
            evidence['navigation_progress'] = estimated_progress
            evidence['plan_progress'] = estimated_progress
            evidence['task_progress'] = estimated_progress
            if dbg:
                print(f"[EVD] progress source: INTERNAL_ESTIMATE  value={round(float(estimated_progress),3)}")
                
                # --- NAV→EXPLORE GRACE: when NAV is effectively done, ease back into EXPLORE ---
        # Detect "NAV finished" by sustained low nav_progress while in NAVIGATE, then for a few
        # steps suppress revisit penalty and cap nav so EXPLORE can win. Also reset NAV quickstart.
        # BEFORE: you’ve already filled `evidence` and computed movement_metrics
        self._apply_post_nav_grace(evidence, movement_metrics=movement_metrics, dbg=True)
        
        
                # --- One-shot NAV quickstart on revisit while we are in EXPLORE ---
        # Fire once per revisit event; we disarm if no revisit this step or we leave EXPLORE.
                # --- Growing NAV quickstart while in EXPLORE and revisit evidence persists ---
        # Accumulates across steps (capped). Resets when we leave EXPLORE.
        try:
            mode = getattr(self, 'current_mode', 'EXPLORE')
        except Exception:
            mode = 'EXPLORE'
        in_explore = (mode == 'EXPLORE')

        kr  = float(evidence.get('known_revisit_rate', 0.0))            # now "intensity"
        rks = bool(evidence.get('recent_known_switch', False))

        # Config knobs
        step_bump = float(getattr(self, 'revisit_quickstart_step', 0.05))   # base growth per step in EXPLORE w/ revisit
        kr_gain   = float(getattr(self, 'revisit_quickstart_gain', 0.40))   # extra growth scaled by intensity
        cap       = float(getattr(self, 'revisit_quickstart_cap',  0.55))   # ceiling

        accum = float(getattr(self, '_rev_quick_accum', 0.0))

        if in_explore:
            if rks or kr > 0.0:
                # grow while we keep seeing revisit evidence
                before_accum = accum
                accum = min(cap, accum + step_bump + kr_gain * kr)
                before = float(evidence.get('navigation_progress', 0.0))
                evidence['navigation_progress'] = max(before, accum)
                if dbg:
                    print(f"[EVD] quickstart NAV(grow): mode=EXPLORE  accum {before_accum:.3f}→{accum:.3f}  "
                          f"kr={kr:.3f} step_bump={step_bump:.2f} kr_gain={kr_gain:.2f} cap={cap:.2f}  "
                          f"nav_prog {before:.3f}→{evidence['navigation_progress']:.3f}")
            # else: no new revisit signal → keep accum as-is (no growth)
        else:
            # left EXPLORE → reset accumulator
            accum = 0.0

        self._rev_quick_accum = accum

        # --- RECOVER bail-out with EXPLORE pivot ---

        try:
            mode = getattr(self, 'current_mode', 'EXPLORE')
        except Exception:
            mode = 'EXPLORE'
        in_recover = (mode == 'RECOVER')

        # Config knobs (override on the instance if desired)
        bail_after    = int(getattr(self, 'recover_bail_after', 1))         # grace steps in RECOVER
        nav_phase     = int(getattr(self, 'recover_nav_phase_steps', 1))     # how many RECOVER steps we try NAV kick
        drop_loop_at  = int(getattr(self, 'recover_drop_loop_at', 3))        # at/after this age: force loop=False

        # NAV-leaning phase (early)
        nav_bump      = float(getattr(self, 'recover_nav_bump', 0.22))       # per-step push toward NAV
        nav_cap       = float(getattr(self, 'recover_nav_cap', 0.55))        # cap for nav_progress during bail
        stag_cap_nav  = float(getattr(self, 'recover_stag_cap_nav', 0.60))   # stagnation clamp while nudging NAV
        ig_floor_nav  = float(getattr(self, 'recover_min_info_gain_nav', 0.20))

        # EXPLORE-leaning phase (late)
        stag_cap_exp  = float(getattr(self, 'recover_stag_cap_explore', 0.45))  # stronger clamp to favor EXPLORE
        ig_floor_exp  = float(getattr(self, 'recover_min_info_gain_explore', 0.35))

        if in_recover:
            age = int(getattr(self, '_recover_age', 0)) + 1
            self._recover_age = age

            # After a short grace, try a brief NAV kick (1 step by default)
            if age >= bail_after and age < bail_after + nav_phase:
                before_nav = float(evidence.get('navigation_progress', 0.0))
                evidence['navigation_progress'] = min(1.0, max(before_nav, min(nav_cap, before_nav + nav_bump)))

                if 'stagnation' in evidence:
                    evidence['stagnation'] = min(float(evidence['stagnation']), stag_cap_nav)
                if 'info_gain' in evidence:
                    evidence['info_gain'] = max(float(evidence['info_gain']), ig_floor_nav)

                if dbg:
                    print(f"[EVD] RECOVER-bail NAV phase: age={age} "
                          f"nav→{evidence['navigation_progress']:.2f}  "
                          f"stag≤{evidence['stagnation']:.2f}  ig≥{evidence['info_gain']:.2f}")

            # If still stuck in RECOVER, pivot to EXPLORE (make EXPLORE win on emissions)
            elif age >= bail_after + nav_phase:
                # Optionally drop the loop penalty after a few steps so RECOVER loses its edge
                if age >= drop_loop_at:
                    evidence['loop_detected'] = False

                # Lower stagnation more aggressively and raise info_gain
                if 'stagnation' in evidence:
                    evidence['stagnation'] = min(float(evidence['stagnation']), stag_cap_exp)
                if 'info_gain' in evidence:
                    evidence['info_gain'] = max(float(evidence['info_gain']), ig_floor_exp)

                # Recompute exploration_productivity to reflect boosted IG and reduced stagnation
                try:
                    expl_factor = float(movement_metrics.get('exploration_factor', 0.5))
                except Exception:
                    expl_factor = 0.5
                evidence['exploration_productivity'] = (
                    float(evidence['info_gain']) * (1.0 - float(evidence['stagnation'])) * expl_factor
                )

                if dbg:
                    print(f"[EVD] RECOVER-bail EXPLORE pivot: age={age} "
                          f"stag≤{evidence['stagnation']:.2f}  ig≥{evidence['info_gain']:.2f}  "
                          f"expl_prod→{evidence['exploration_productivity']:.2f}  "
                          f"loop={evidence.get('loop_detected', None)}")

        else:
            # reset counter when we leave RECOVER
            self._recover_age = 0
                # ---------- TASK gate (external flag only) ----------
        # --- Map coverage as a separate input: arm an external-style task flag ---
        cov = None
        try:
            cov, done = self._update_task_flag_from_env()
        except Exception:
            done = False

        if cov is not None:
            evidence['room_coverage'] = float(cov)    # for telemetry; not used in emissions directly
        evidence['_rooms_complete'] = bool(done)      # ditto

        # --- Decoupled task gate: "external flag only" (what you asked for) ---
        require_flag = bool(getattr(self, 'task_require_external_flag', True))
        armed       = bool(getattr(self, 'task_external_ready', False))  # set by _update_task_flag_from_env()

        if require_flag and not armed:
            # keep task_progress silent until the flag is set
            evidence['task_progress'] = 0.0
        else:
            # when armed, let task progress track plan progress
            evidence['task_progress'] = float(evidence.get('plan_progress', 0.0))
        if dbg:
            print(f"[EVD] task gate (flag): require={require_flag} ready={armed} "
                  f"task_progress={evidence['_rooms_complete'] :.2f} armed={evidence['room_coverage']}")
        # When armed, softly mute NAV so TASK wins emissions in the HMM,
        # and (optionally) lower exploration productivity a bit to avoid EXPLORE re-winning.
        if armed:
            nav_cap   = float(getattr(self, 'task_nav_cap_when_armed', 0.20))  # cap NAV progress
            task_floor= float(getattr(self, 'task_progress_floor_when_armed', 0.60))  # make TASK attractive
            evidence['navigation_progress'] = min(float(evidence.get('navigation_progress', 0.0)), nav_cap)
            evidence['task_progress']       = max(float(evidence.get('task_progress', 0.0)), task_floor)

            damp_explore = bool(getattr(self, 'task_damp_explore_when_armed', True))
            if damp_explore:
                evidence['exploration_productivity'] = 0.8 * float(evidence.get('exploration_productivity', 0.0))

        # Recovery effectiveness using centralized metrics (agent_lost passed for gating inside the function)
        evidence['recovery_effectiveness'] = self._calculate_recovery_effectiveness_centralized(
            doubt_metrics, movement_metrics, position_metrics, evidence['agent_lost']
        )
        if dbg:
            print(f"[EVD] recovery_effectiveness={round(float(evidence['recovery_effectiveness']),3)}  "
                f"(agent_lost={evidence['agent_lost']})")

        # New node detection
        evidence['new_node_created'] = self._detect_new_node_created_centralized(recent_entries)
        if dbg:
            print(f"[EVD] new_node_created={bool(evidence['new_node_created'])}")

        if dbg:
            # Final evidence snapshot (ordered print)
            keys_order = ['agent_lost', 'loop_detected',
                        'movement_score', 'stagnation',
                        'info_gain', 'exploration_productivity',
                        'navigation_progress', 'plan_progress', 'task_progress',
                        'recovery_effectiveness', 'new_node_created']
            snapshot = {k: evidence.get(k) for k in keys_order}
            print("[EVD] evidence(final):", snapshot)

        return evidence


    def _calculate_movement_metrics(self, poses, positions):
        """Calculate all movement-related metrics in one place"""
        metrics = {
            'movement_score': 0.5,
            'total_distance': 0.0,
            'movement_variance': 0.0,
            'exploration_factor': 0.5
        }
        
        if len(poses) < 2:
            return metrics
        print("POSESSS",poses,positions)
        # Calculate total distance moved
        total_distance = 0.0
        distances = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dist = (dx**2 + dy**2)**0.5
            total_distance += dist
            distances.append(dist)
        
        metrics['total_distance'] = total_distance
        
        # Movement score (normalized)
        metrics['movement_score'] = min(total_distance / 10.0, 1.0)
        
        # Movement variance (how varied are the movements)
        if len(distances) > 1:
            mean_dist = np.mean(distances)
            variance = np.var(distances)
            metrics['movement_variance'] = min(variance / (mean_dist + 1e-6), 1.0)
        
        # Exploration factor (combination of movement and variance)
        metrics['exploration_factor'] = (
            0.7 * metrics['movement_score'] + 
            0.3 * metrics['movement_variance']
        )
        
        return metrics
    
    def _calculate_revisit_metrics(self, recent_entries, positions):
        """
        Revisit detection with global node memory (v5-intensity) + NEW 'new-node grace':
        • If the current transition enters a truly new node (visits_prior == 0),
            suppress the revisit penalty for a short TTL (grace).
        • Grace continues while dwelling in that new node and refreshes if you
            keep chaining into other new nodes.
        • Entering a known node cancels grace immediately (penalty resumes).

        Return keys remain the same:
        {'recent_known_switch': bool, 'known_revisit_rate': float in [0,1]}
        """
        import collections, math

        # ---- original knobs (unchanged) ----
        min_gap   = int(getattr(self, 'revisit_min_gap', 10))
        win_trans = int(getattr(self, 'revisit_window', 60))
        alpha     = float(getattr(self, 'revisit_weight_alpha', 0.25))
        gamma     = float(getattr(self, 'revisit_decay_gamma', 0.90))
        beta      = float(getattr(self, 'revisit_intensity_beta', 0.60))

        # ---- NEW knobs (override on instance if desired) ----
        # How many evidence steps to keep revisit penalty softened after entering a new node.
        newnode_grace_steps = int(getattr(self, 'new_node_revisit_free_steps', 10))
        # Multiply the intensity during grace (0.0 = fully suppress; 0.25 = soften)
        newnode_grace_scale = float(getattr(self, 'new_node_revisit_free_scale', 0.0))
        # -----------------------------------------------------

        # ---------- lazy init of persistent state ----------
        if not hasattr(self, '_rev_step'):               self._rev_step = 0
        if not hasattr(self, '_rev_last_id'):            self._rev_last_id = None
        if not hasattr(self, '_rev_last_seen_step'):     self._rev_last_seen_step = {}
        if not hasattr(self, '_rev_visit_count'):        self._rev_visit_count = {}
        if not hasattr(self, '_rev_recent_transitions'): self._rev_recent_transitions = collections.deque(maxlen=max(8, win_trans))
        # NEW: track whether we are currently on a node that was first entered as "new",
        # and how many steps of grace remain.
        if not hasattr(self, '_rev_curr_node_is_new'):   self._rev_curr_node_is_new = False
        if not hasattr(self, '_rev_new_grace_left'):     self._rev_new_grace_left = 0
        # ---------------------------------------------------

        # ---- pull the current node_id from the newest entry ----
        def _get_id(e):
            for k in ('node_id', 'place_id', 'exp_id'):
                if k in e and e[k] is not None:
                    return e[k]
            meta = e.get('meta') or e.get('info') or {}
            if isinstance(meta, dict):
                for k in ('node_id', 'place_id', 'exp_id'):
                    if meta.get(k) is not None:
                        return meta[k]
            return None

        def _as_int(v):
            try:
                if hasattr(v, "item"): v = v.item()
                return int(v)
            except Exception:
                try:
                    import numpy as _np
                    return int(_np.asarray(v).item())
                except Exception:
                    return None

        curr_id = None
        if recent_entries:
            curr_id = _as_int(_get_id(recent_entries[-1]))

        # advance global step counter (one tick per evidence extraction)
        step = self._rev_step
        self._rev_step = step + 1

        last_id = self._rev_last_id
        recent_known_switch = False

        # Track whether THIS step entered a brand-new node
        entered_new_node_this_step = False

        if curr_id is None:
            # no node this step
            tail = list(self._rev_recent_transitions)
        else:
            if last_id is None:
                # first ever node
                self._rev_visit_count[curr_id] = self._rev_visit_count.get(curr_id, 0) + 1
                self._rev_last_seen_step[curr_id] = step
                self._rev_last_id = curr_id
                # First ever counts as new
                entered_new_node_this_step = True
                self._rev_curr_node_is_new = True
                self._rev_new_grace_left = max(self._rev_new_grace_left, newnode_grace_steps)
            elif curr_id != last_id:
                # transition last_id -> curr_id
                visits_prior = self._rev_visit_count.get(curr_id, 0)      # before this entry
                last_seen    = self._rev_last_seen_step.get(curr_id, None) # historical last time at curr_id
                is_revisit   = (last_seen is not None) and ((step - last_seen) >= min_gap)
                # record (for rolling intensity)
                self._rev_recent_transitions.append((bool(is_revisit), int(curr_id), int(visits_prior)))
                # update counters
                self._rev_visit_count[curr_id] = visits_prior + 1
                self._rev_last_seen_step[curr_id] = step
                self._rev_last_seen_step[last_id] = step - 1
                self._rev_last_id = curr_id

                if visits_prior == 0:
                    # NEW node: start/refresh grace and mark current node as new
                    entered_new_node_this_step = True
                    self._rev_curr_node_is_new = True
                    self._rev_new_grace_left = max(self._rev_new_grace_left, newnode_grace_steps)
                else:
                    # KNOWN node: cancel grace immediately; penalty should apply
                    self._rev_curr_node_is_new = False
                    self._rev_new_grace_left = 0

                recent_known_switch = bool(is_revisit)
            else:
                # dwell on the same node -> keep last-seen fresh
                self._rev_last_seen_step[curr_id] = step
            tail = list(self._rev_recent_transitions)

        # ----- original decayed intensity mapped to [0,1] (unchanged) -----
        if not tail:
            intensity_raw = 0.0
        else:
            L = len(tail)
            sum_w = 0.0
            for i, (flag, node, visits_before) in enumerate(tail):
                if not flag:
                    continue
                age = (L - 1 - i)                       # newer transitions have smaller age
                prior_boost = (1.0 + alpha * max(0, visits_before - 1))
                sum_w += (gamma ** age) * prior_boost
            intensity_raw = float(1.0 - math.exp(-beta * sum_w))

        # ----- NEW: suppress/soften intensity while we are on / just entered a NEW node -----
        intensity = intensity_raw
        if self._rev_curr_node_is_new and self._rev_new_grace_left > 0:
            intensity = float(intensity_raw) * float(newnode_grace_scale)
            self._rev_new_grace_left -= 1
            # If grace expired while dwelling, drop the 'curr_node_is_new' flag so normal penalty resumes next step
            if self._rev_new_grace_left <= 0:
                self._rev_curr_node_is_new = False

        if getattr(self, 'debug_evidence', False):
            dbg_tail = tail[-min(10, len(tail)):]
            dbg_flags  = [f for (f, n, vb) in dbg_tail]
            dbg_nodes  = [n for (f, n, vb) in dbg_tail]
            dbg_visits = [vb for (f, n, vb) in dbg_tail]
            print(f"[EVD] revisit[v6-intensity+newnode-grace]: "
                f"last_id={last_id} -> curr_id={curr_id}  recent_switch={recent_known_switch}  "
                f"raw={intensity_raw:.2f} → out={intensity:.2f}  "
                f"min_gap={min_gap}  tail_flags={dbg_flags} nodes={dbg_nodes} visits_before={dbg_visits}  "
                f"entered_new={entered_new_node_this_step} curr_new={self._rev_curr_node_is_new} "
                f"grace_left={self._rev_new_grace_left}")

        return {
            'recent_known_switch': bool(recent_known_switch),
            'known_revisit_rate':  float(intensity),
        }


    def _calculate_position_metrics(self, positions, recent_entries):
        """Calculate all position-related metrics in one place"""
        metrics = {
            'position_repetition': 0.0,
            'position_diversity': 0.5,
            'position_stagnation': False,
            'unique_positions': 0
        }
        
        if len(positions) < 3:
            return metrics
        
        # Round positions to avoid floating point issues
        rounded_positions = [(round(pos[0], 1), round(pos[1], 1)) for pos in positions]
        
        # Calculate position repetition
        position_counts = {}
        for pos in rounded_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        if position_counts:
            max_count = max(position_counts.values())
            metrics['position_repetition'] = max_count / len(positions)
            metrics['unique_positions'] = len(position_counts)
            
            # Position diversity (inverse of repetition, adjusted)
            metrics['position_diversity'] = 1.0 - (metrics['position_repetition'] - 1.0/len(positions))
            metrics['position_diversity'] = max(0.0, min(1.0, metrics['position_diversity']))
        
        # Position stagnation check
        if len(recent_entries) >= 8:
            last_4_positions = rounded_positions[-4:]
            if len(set(last_4_positions)) <= 2:  # Only 1-2 unique positions in last 4 steps
                metrics['position_stagnation'] = True
        
        return metrics

    def _calculate_doubt_metrics(self, doubt_counts):
        """Calculate all doubt-related metrics in one place"""
        metrics = {
            'current_doubt': 0,
            'doubt_trend': 0.0,
            'max_doubt': 0,
            'doubt_stability': 0.5
        }
        
        if not doubt_counts:
            return metrics
        
        metrics['current_doubt'] = doubt_counts[-1]
        metrics['max_doubt'] = max(doubt_counts)
        
        # Calculate trend (simple linear regression slope)
        if len(doubt_counts) >= 3:
            x = np.arange(len(doubt_counts))
            y = np.array(doubt_counts)
            if len(x) > 1 and np.std(y) > 1e-6:
                slope, _ = np.polyfit(x, y, 1)
                metrics['doubt_trend'] = slope
        
        # Doubt stability (how consistent are the doubt levels)
        if len(doubt_counts) > 1:
            metrics['doubt_stability'] = 1.0 / (1.0 + np.std(doubt_counts))
        
        return metrics

    def _determine_agent_lost(self, current_doubt_count, movement_metrics, position_metrics):
        """Determine if agent is lost using centralized metrics"""
        base_threshold = 6
        
        if current_doubt_count <= base_threshold:
            return False
        
        # High repetition indicates being lost
        if position_metrics['position_repetition'] > 0.6:
            return True
        
        # Low movement + very high doubt
        if (movement_metrics['movement_variance'] < 0.2 and 
            current_doubt_count > base_threshold + 2):
            return True
        
        # Extremely high doubt regardless
        if current_doubt_count > base_threshold + 4:
            return True
        
        # Position stagnation + moderate doubt
        if (position_metrics['position_stagnation'] and 
            current_doubt_count > 3):
            return True
        
        return False

    def _determine_loop_detected(self, position_metrics, buffer_list):
        """Determine if loop is detected using centralized metrics"""
        if len(buffer_list) < self.params['loop_threshold']:
            return False
        
        # Use position repetition as primary indicator
        if position_metrics['position_repetition'] > 0.3:
            return True
        
        # Additional check: recent position clustering
        recent_positions = []
        for entry in buffer_list[-self.params['loop_threshold']:]:
            if 'real_pose' in entry:
                pos = entry['real_pose']
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    recent_positions.append((round(pos[0], 1), round(pos[1], 1)))
        
        if len(recent_positions) >= self.params['loop_threshold']:
            unique_recent = len(set(recent_positions))
            repetition_ratio = 1.0 - (unique_recent / len(recent_positions))
            return repetition_ratio > 0.4
        
        return False

    def _calculate_recovery_effectiveness_centralized(self, doubt_metrics, movement_metrics, position_metrics,agent_lost):
        """Calculate recovery effectiveness using centralized metrics"""
        if doubt_metrics['current_doubt'] == 0:
            # No recovery needed; keep this low so RECOVER isn't rewarded
            return 0.1
         # When not lost, do NOT reward recovery; this prevents RECOVER from winning spuriously.
        if not agent_lost:
            return 0.10  # baseline low; tune if needed
        recovery_score = 0.5  # Base score
        
        # Doubt trend analysis
        if doubt_metrics['doubt_trend'] < -0.5:  # Strong decreasing trend
            recovery_score += 0.3
        elif doubt_metrics['doubt_trend'] < 0:  # Mild decreasing trend
            recovery_score += 0.1
        elif doubt_metrics['doubt_trend'] > 0.5:  # Increasing trend (getting worse)
            recovery_score -= 0.3
        
        # Movement analysis
        if movement_metrics['movement_score'] > 0.3:  # Good movement
            recovery_score += 0.1
        elif movement_metrics['movement_score'] < 0.1:  # Too little movement
            recovery_score -= 0.1
        
        # Position diversity helps recovery
        if position_metrics['position_diversity'] > 0.4:
            recovery_score += 0.1
        
        # Avoid repetitive behavior during recovery
        if position_metrics['position_repetition'] > 0.7:
            recovery_score -= 0.2
        
        return max(0.0, min(1.0, recovery_score))

    def _estimate_progress_from_movement(self, movement_metrics, doubt_metrics, position_metrics):
        """Estimate progress when external progress info is not available"""
        base_progress = 0.3
        
        # Good movement suggests progress
        if movement_metrics['movement_score'] > 0.4:
            base_progress += 0.2
        
        # Low doubt suggests good progress
        if doubt_metrics['current_doubt'] < 3:
            base_progress += 0.2
        elif doubt_metrics['current_doubt'] > 8:
            base_progress -= 0.2
        
        # Position diversity suggests exploration progress
        if position_metrics['position_diversity'] > 0.5:
            base_progress += 0.1
        
        # Decreasing doubt trend suggests recovery progress
        if doubt_metrics['doubt_trend'] < -0.3:
            base_progress += 0.2
        
        return max(0.0, min(1.0, base_progress))

    def _detect_new_node_created_centralized(self, recent_entries):
        """Detect new node creation using centralized approach"""
        if len(recent_entries) < 2:
            return False
        
        # Check for node_id changes in recent entries
        node_ids = []
        for entry in recent_entries[-5:]:  # Check last 5 steps
            if 'node_id' in entry:
                node_ids.append(entry['node_id'])
        
        if len(node_ids) >= 2:
            # If node_id changed recently, likely a new node was created
            return len(set(node_ids)) > 1
        
        return False
    
        # ───────────────────────────────────────────────────────────────────
    #  0.  Small helper for consistent, compact arrays in prints
    # ───────────────────────────────────────────────────────────────────
    def _vec2str(self, v):
        """Return first k elements of a vector as a string."""
        v = np.asarray(v)
        k=len(v)
        head = ", ".join(f"{x:.3f}" for x in v[:k])
        suffix = "…" if v.size > k else ""
        return f"[{head}{suffix}]"

    # ───────────────────────────────────────────────────────────────────
    #  1.  Prior likelihood  (uniform over joint states)
    # ───────────────────────────────────────────────────────────────────
    def _compute_prior_likelihood(self, evidence: Dict) -> float:
        likes = [
            self.compute_emission_likelihood(evidence, state)
            for state in self.hierarchical_states
        ]
        prior_like = float(np.mean(likes))
        print(f"[DBG-PRIOR]   evidence={evidence}  "
            f"mean emisLike={prior_like:.3e}")
        return prior_like

    # ───────────────────────────────────────────────────────────────────
    #  2.  Joint weighted likelihood (soft mixture)
    # ───────────────────────────────────────────────────────────────────
    def _compute_joint_weighted_likelihood(self, evidence: Dict, var: str) -> float:
        total = 0.0
        for idx, state in enumerate(self.hierarchical_states):
            p_state = self.state_beliefs[idx]
            prm = self.emission_params[state].get(var)
            if isinstance(prm, dict):
                pdf = stats.norm.pdf(evidence[var], prm['mean'], prm['std'])
                total += p_state * pdf
        print(f"[DBG-MIX]     var={var:<23} mixedPDF={total:.3e}")
        return total

    # ───────────────────────────────────────────────────────────────────
    #  3.  Run-length-specific likelihood for BOCPD
    # ───────────────────────────────────────────────────────────────────
    def compute_run_length_specific_likelihood(self, evidence: Dict,
                                            run_length: int) -> float:
        if run_length == 0:
            return self._compute_prior_likelihood(evidence)

        hist = list(self.evidence_history)
        relevant = hist[-run_length:] if len(hist) >= run_length else hist
        print(f"[DBG-RL-{run_length:02d}] ---- scoring run length {run_length} "
        f"using {len(relevant)} past samples ----")

        if not relevant:
            return self._compute_prior_likelihood(evidence)

        logl = 0.0
        variables_to_check = ['Δ_info_gain', 'Δ_stagnation', 'Δ_movement_score',
                    'Δ_exploration_productivity', 'Δ_navigation_progress',
                    'Δ_recovery_effectiveness', 'Δ_plan_progress', 'Δ_task_progress'
                ]
        for var in variables_to_check:

            if var in evidence:
                vals = [e[var] for e in relevant if var in e]
                if len(vals) >= 2:
                    μ = np.mean(vals)
                    σ = max(np.std(vals), 0.05) * (1 + 1.0/run_length)
                    lp = stats.norm.logpdf(evidence[var], μ, σ)
                    logl += lp
                    print(f"[DBG-RL-{run_length:02d}] var={var:<23} μ={μ:.3f} "
                        f"σ={σ:.3f}  logp={lp:.3f}")
                else:
                    mix = self._compute_joint_weighted_likelihood(evidence, var)
                    lp  = np.log(mix + 1e-10)
                    logl += lp
                    print(f"[DBG-RL-{run_length:02d}] var={var:<23} "
                        f"fallback logp={lp:.3f}")

        # static emission mixture
        emis = np.array([
            self.compute_emission_likelihood(evidence, s)
            for s in self.hierarchical_states
        ])
        static_mix = float(np.dot(self.state_beliefs, emis))
        comb = 0.7 * np.exp(logl) + 0.3 * static_mix
        print(f"[DBG-RL-{run_length:02d}] combined like={comb:.3e}",static_mix,comb)
        return max(comb, 1e-10)

    # ───────────────────────────────────────────────────────────────────
    #  4.  BOCPD update
    def bocpd_update(self, evidence: Dict) -> Tuple[bool, float]:
        """
        Update run-length distribution p(r_t | e_1:t) using Adams & MacKay’s
        BOCPD, but:
            • work in log-space first        (avoids under/overflow)
            • derive a **proper probability** of changepoint (0–1)
        Returns
        -------
        changepoint_detected : bool      True if cp_prob crosses threshold
        cp_prob              : float     p(r_t = 0 | evidence)
        """

        # 1) log-likelihoods  ℓ_r = log p(e_t | r_{t-1}=r)
        log_like = np.full(self.max_run_length + 1, -np.inf)
        for r in range(len(self.run_length_dist)):               # only where mass
            if self.run_length_dist[r] > 1e-10:
                l = self.compute_run_length_specific_likelihood(evidence, r)
                log_like[r] = np.log(max(l, 1e-10))              # clamp at 1e-10
                print(f"[DBG-BAYES-LL] r={r:02d} raw_like={l:.3e} "
                        f"log_like={log_like[r]:.3f}")

        # 2) stabilise & exponentiate back to ordinary scale
        max_log = np.max(log_like)
        if max_log == -np.inf:                 # <-- add this block
            like = np.ones_like(log_like)      # uniform – no information
        else:
            like = np.exp(log_like - max_log)
        print(f"[DBG-MAX] max_log={max_log:.3f}")                       # safe, all ≤ 1
        print(f"[DBG-LIKE] like[:10]={like[:10]}")
        # 3) “growth” term  p(r_t=r+1 , no-cp)
        growth = np.zeros_like(self.run_length_dist)
        if len(self.run_length_dist) > 1:
            growth[1:] = (self.run_length_dist[:-1] *
                        (1.0 - self.hazard_rate) *
                        like[:-1])

        # 4) “changepoint” numerator  p(r_t=0 , cp)
        cp_num = np.sum(self.run_length_dist * self.hazard_rate * like)
        print(f"[DBG-GROW] growth[:5]={growth[:5]}  cp_num={cp_num:.3e}")
        

        # 5) evidence normaliser  p(e_t | e_1:t-1)
        evidence_prob = cp_num + np.sum(growth[1:])
        print(f"[DBG-NORM] evidence_prob={evidence_prob:.3e}")
        # 6) proper posterior run-length distribution
        new_dist = np.zeros_like(self.run_length_dist)
        new_dist[0] = cp_num
        new_dist[1:] = growth[1:]
        if evidence_prob > 0:
            den = max(evidence_prob, 1e-12)
            new_dist /= den

        else:                                   # pathological case
            new_dist[:] = 1.0 / len(new_dist)

        self.run_length_dist = new_dist
        cp_prob = new_dist[0]                  # now guaranteed ∈ [0,1]

        changep = cp_prob > self.changepoint_threshold
        print(f"[DBG-BOCPD] cp_prob={cp_prob:.3f}  changepoint={changep}  "
            f"runLenDist(0..4)={new_dist}")

        return changep, cp_prob
    def get_mode_probabilities(self) -> Dict[str, float]:
        """Belief mass per mode (since states == modes)."""
        return {m: float(self.state_beliefs[self.state_index[m]]) for m in self.states}

    def get_most_likely_state(self) -> str:
        """Return the most probable mode."""
        self.current_mode=self.states[int(np.argmax(self.state_beliefs))]
        print("CURRENT MODE",self.current_mode)
        return self.states[int(np.argmax(self.state_beliefs))]

    def compute_state_entropy(self) -> float:
        """Shannon entropy of the state belief."""
        p = np.clip(self.state_beliefs, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())

    def compute_mode_entropy(self) -> float:
        """Alias to state entropy (states==modes)."""
        return self.compute_state_entropy()
    # ───────────────────────────────────────────────────────────────────
    #  5.  HMM forward step
    # ───────────────────────────────────────────────────────────────────
    def hmm_forward_step(self, evidence: Dict) -> np.ndarray:
        e_likes = np.array([self.compute_emission_likelihood(evidence, m) for m in self.states], dtype=float)
        if not np.all(np.isfinite(e_likes)) or e_likes.sum() <= 0:
            e_likes = np.full(self.n_states, 1e-9, dtype=float)

        pred = self.transition_matrix.T @ self.state_beliefs
        if not np.all(np.isfinite(pred)) or pred.sum() <= 0:
            pred = np.ones(self.n_states, dtype=float) / self.n_states

        post = e_likes * pred
        s = float(post.sum())
        self.state_beliefs = post / s if s > 0 else np.ones(self.n_states, dtype=float) / self.n_states

        print(f"[DBG-HMM]    eLikes={self._vec2str(e_likes)}  pred={self._vec2str(pred)}  newBelief={self._vec2str(self.state_beliefs)}")
        return self.state_beliefs
    # ───────────────────────────────────────────────────────────────────
    #  6.  GLOBAL UPDATE ENTRY POINT
    # ───────────────────────────────────────────────────────────────────
    def update(self, replay_buffer, external_info_gain=None,
           external_plan_progress=None) -> Tuple[np.ndarray, Dict]:
        """Evidence → (optional deltas) → gates → T → (optional) BOCPD → HMM forward → diagnostics."""

        t0 = time.perf_counter()
        self.step_counter = getattr(self, 'step_counter', 0) + 1
        print(f"\n================ STEP {self.step_counter} ================")

        # 1) Evidence
        evidence = self.extract_evidence_from_replay_buffer(
            replay_buffer, external_info_gain, external_plan_progress
        )
        print("[UPD:1] Evidence:", evidence)

        evidence2 = self._add_deltas(evidence)
        if evidence2 is evidence:
            print("[UPD:1] Δ-features: primed next step (first frame, no deltas yet).")
        else:
            dkeys = [k for k in evidence2.keys() if k.startswith('Δ_')]
            print(f"[UPD:1] Δ-features added: {dkeys}")

        self.evidence_buffer.append(evidence)
        self.evidence_history.append(evidence.copy())
        self.state_history.append(self.state_beliefs.copy())

        # 2) Task progress smoothing + optional gate fn
        tp_now = float(evidence.get('task_progress', evidence.get('plan_progress', 0.0)))
        self.task_progress_smooth = 0.9 * self.task_progress_smooth + 0.1 * tp_now
        print(f"[UPD:2] task_progress now={tp_now:.3f}  smooth={self.task_progress_smooth:.3f}")

        used_gate_fn = False
        if callable(getattr(self, 'task_gate_fn', None)):
            try:
                eg, xg = self.task_gate_fn(self, evidence2, self.task_progress_smooth)
                self.task_enter_gate = float(np.clip(eg, 1e-6, 1e6))
                self.task_exit_gate  = float(np.clip(xg, 1e-6, 1e6))
                used_gate_fn = True
            except Exception as e:
                print(f"[UPD:2][WARN] task_gate_fn threw: {e} — keeping prior gates.")

        print(f"[UPD:2] gates: enter={self.task_enter_gate:.4g}  exit={self.task_exit_gate:.4g}  "
            f"source={'task_gate_fn' if used_gate_fn else 'fixed/neutral'}")

        # 3) Transition rebuild
        tT0 = time.perf_counter()
        T = self._build_transition_matrix()
        if not np.all(np.isfinite(T)):
            print("[UPD:3][ERR] Non-finite T; forcing uniform.")
            T = np.ones_like(T) / T.shape[1]
        self.transition_matrix = T
        rows = T.sum(axis=1)
        print(f"[UPD:3] T row sums min/max: {rows.min():.6f}/{rows.max():.6f}  shape={T.shape}  "
            f"time={(time.perf_counter()-tT0)*1e3:.1f} ms")
        print("[UPD:3] T (rows→cols):")
        for i, fm in enumerate(self.states):
            top = sorted([(tm, T[i, self.state_index[tm]]) for tm in self.states],
                        key=lambda x: x[1], reverse=True)
            print(f"         {fm:<12} → " + ", ".join(f"{tm}:{p:0.3f}" for tm,p in top))

        # 4) BOCPD (optional)
        changep, cp_prob = False, 0.0
        if getattr(self, 'use_bocpd', True):
            try:
                # canonical flow: pass dict evidence; expect (bool, float)
                cp_flag, cp_mass = self.bocpd_update(evidence2)
                changep, cp_prob = bool(cp_flag), float(cp_mass)
                print(f"[UPD:4] BOCPD: cp_prob={cp_prob:.3f}  "
                    f"{'TRIGGERED' if changep else '—'}  "
                    f"(hazard={getattr(self, 'hazard_rate', None)})")
            except Exception as e:
                print(f"[UPD:4][ERR] bocpd_update failed: {e} — continuing without CPD.")

        # 5) HMM forward step
        try:
            # Emissions per mode
            e_likes = np.array([self.compute_emission_likelihood(evidence2, m)
                                for m in self.states], dtype=float)
            if not np.all(np.isfinite(e_likes)) or e_likes.sum() <= 0:
                print("[UPD:5][WARN] Bad emission vector; using epsilons.")
                e_likes = np.full(self.n_states, 1e-9, dtype=float)

            pred = self.transition_matrix.T @ self.state_beliefs
            if not np.all(np.isfinite(pred)) or pred.sum() <= 0:
                print("[UPD:5][ERR] Bad prediction; resetting to uniform.")
                pred = np.ones(self.n_states, dtype=float) / self.n_states

            post = e_likes * pred
            s = float(post.sum())
            if s > 0:
                self.state_beliefs = post / s
            else:
                print("[UPD:5][ERR] Posterior sum=0; using uniform belief.")
                self.state_beliefs[:] = 1.0 / self.n_states

            print("[UPD:5] eLikes:", {m: float(e_likes[self.state_index[m]]) for m in self.states})
            print("[UPD:5] pred:  ", {m: float(pred[self.state_index[m]]) for m in self.states})
            print("[UPD:5] post:  ", {m: float(self.state_beliefs[self.state_index[m]]) for m in self.states})

        except Exception as e:
            print(f"[UPD:5][ERR] HMM forward step failed: {e}")
            self.state_beliefs[:] = 1.0 / self.n_states

        # 6) Attempt counters
        if getattr(self, 'prev_evidence', None) is not None:
            lost = bool(evidence.get('agent_lost', False))
            navp = float(evidence.get('navigation_progress', 0.0))
            thr = 0.4
            if lost:
                self.recovery_attempts = getattr(self, 'recovery_attempts', 0) + 1
                print("[UPD:6] counter: RECOVER++ (agent_lost=True)")
            elif navp > thr:
                self.navigation_attempts = getattr(self, 'navigation_attempts', 0) + 1
                print(f"[UPD:6] counter: NAVIGATE++ (navigation_progress={navp:.2f} > {thr})")
            else:
                self.exploration_attempts = getattr(self, 'exploration_attempts', 0) + 1
                print("[UPD:6] counter: EXPLORE++ (default path)")
        self.prev_evidence = evidence

        # 7) Diagnostics payload
        diag = {
            'mode_probabilities': self.get_mode_probabilities(),
            'most_likely_state':  self.get_most_likely_state(),
            'state_entropy':      self.compute_state_entropy(),
            'mode_entropy':       self.compute_mode_entropy(),
            'changepoint_detected': changep,
            'changepoint_probability': cp_prob,
            'run_length_dist':    self.run_length_dist.copy() if hasattr(self, 'run_length_dist') else None,
            'evidence_buffer_size': len(self.evidence_buffer),
            'evidence_history_size': len(self.evidence_history),
            'evidence_used':      evidence2,
            'task_progress':      {'now': tp_now, 'smooth': self.task_progress_smooth},
            'task_gates':         {'enter': self.task_enter_gate, 'exit': self.task_exit_gate},
        }

        print(f"[UPD:∑] wall-time={(time.perf_counter()-t0)*1e3:.1f} ms  "
            f"| belief.shape={self.state_beliefs.shape}  T.shape={self.transition_matrix.shape}")

        return self.state_beliefs, diag

    
    def _add_deltas(self, evidence: dict) -> dict:
        """
        Return a *new* dict that contains the original evidence **plus**
        first-difference features Δ_x = x_t – x_{t-1} for every scalar
        we care about.  Uses self.prev_evidence for the t-1 values.
        """
        keys = [
            'info_gain', 'stagnation', 'movement_score',
            'exploration_productivity', 'navigation_progress',
            'recovery_effectiveness', 'plan_progress', 'task_progress',
        ]

        if self.prev_evidence is None:                # first frame → no deltas yet
            self.prev_evidence = evidence.copy()
            return evidence

        # build a shallow copy so we don’t mutate the caller’s dict
        aug = evidence.copy()

        for k in keys:
            if k in evidence and k in self.prev_evidence:
                aug[f'Δ_{k}'] = evidence[k] - self.prev_evidence[k]

        self.prev_evidence = evidence.copy()          # store for next call
        return aug
    def _mode_stickiness(self, mode: str) -> float:
        """
        Probability of staying in the same mode. Decays for non-task modes as
        task progress rises; slight bonus for TASK_SOLVING.
        """
        base = {
            'EXPLORE':      0.92,
            'NAVIGATE':     0.93,
            'RECOVER':      0.10,
            'TASK_SOLVING': 0.95,
        }.get(mode, 0.90)

        tp = float(getattr(self, 'task_progress_smooth', 0.0))
        if mode == 'TASK_SOLVING':
            val = min(base + self.stickiness_task_bonus * tp, 0.995)
        else:
            val = max(base * (1.0 - self.stickiness_decay_non_task * tp), 0.50)

        return float(np.clip(val, 0.05, 0.995))
    
    
    def compute_state_entropy(self) -> float:
        """Compute entropy over the full state space"""
        probs = self.state_beliefs + 1e-10  # Avoid log(0)
        return -np.sum(probs * np.log(probs))
    
    def compute_mode_entropy(self) -> float:
        """Compute entropy over just the mode space"""
        mode_probs = list(self.get_mode_probabilities().values())
        mode_probs = np.array(mode_probs) + 1e-10
        return -np.sum(mode_probs * np.log(mode_probs))
    
    def get_strategy_recommendation(self) -> Dict:
        """Concrete strategy recommendation without submodes."""
        most_likely_mode = self.get_most_likely_state()
        mode_probs = self.get_mode_probabilities()
        return {
            'recommended_mode': most_likely_mode,
            'mode_confidence': mode_probs[most_likely_mode],
            'mode_probabilities': mode_probs,
            'uncertainty': self.compute_state_entropy(),
            'changepoint_mass': self.run_length_dist[0],
            'most_likely_run_length': int(np.argmax(self.run_length_dist)),
        }

    

    # Complete the get_diagnostics method (the end was cut off)
    def get_diagnostics(self) -> Dict:
        """
        Return comprehensive diagnostic information about the HHMM+BOCPD state.
        """
        current_mode_probs = self.get_mode_probabilities()
        dominant_mode = max(current_mode_probs.items(), key=lambda x: x[1])
        
        most_likely_state = self.get_most_likely_state()
        
        return {
            'state_beliefs': self.state_beliefs.copy(),
            'mode_probabilities': current_mode_probs,
            'dominant_mode': dominant_mode[0],
            'dominant_mode_probability': dominant_mode[1],
            'most_likely_state': most_likely_state,
            'state_entropy': self.compute_state_entropy(),
            'mode_entropy': self.compute_mode_entropy(),
            'run_length_distribution': self.run_length_dist.copy(),
            'changepoint_mass': self.run_length_dist[0],
            'most_likely_run_length': np.argmax(self.run_length_dist),
            'evidence_buffer_length': len(self.evidence_buffer),
            'state_history_length': len(self.state_history),
            'stagnation_counter': self.stagnation_counter,
            'exploration_attempts': self.exploration_attempts,
            'navigation_attempts': self.navigation_attempts,
            'recovery_attempts': self.recovery_attempts
        }

# Add the HierarchicalBayesianController class
class HierarchicalBayesianController:
    """
    Controller that integrates the TrueHierarchicalHMMWithBOCPD for 
    adaptive agent behavior management with changepoint detection.
    """
    
    def __init__(self, key=None):
        self.hhmm = TrueHierarchicalHMMWithBOCPD(use_submodes=False, use_bocpd=False)
        self.strategy_history = deque(maxlen=100)
        self.last_update_time = time.time()
        
        # Remove reward-based tracking
        self.step_count = 0
        self.successful_transitions = 0
        self.total_transitions = 0

    def update(self, replay_buffer, external_info_gain=None, external_plan_progress=None):
        """
        Updated controller update method
        
        Args:
            replay_buffer: External replay buffer with agent state history
            info_gain_func: Optional function that returns info_gain value
            plan_progress_func: Optional function that returns plan progress value
        """
        # Update hierarchical HMM with BOCPD
        beliefs, diagnostics = self.hhmm.update(
            replay_buffer, external_info_gain, external_plan_progress
        )

        # Get strategy recommendation
        strategy = self.hhmm.get_strategy_recommendation()

        # Store strategy history
        self.strategy_history.append({
            'timestamp': time.time(),
            'strategy': strategy,
            'evidence': diagnostics.get('evidence_used', {}),
            'replay_buffer_size': len(replay_buffer)
        })

        self.step_count += 1
        self.last_update_time = time.time()

        return strategy, diagnostics

    # Remove reward-related methods and update remaining ones accordingly
    def get_summary_stats(self) -> Dict:
        """Updated summary stats without reward tracking"""
        if not self.strategy_history:
            return {'status': 'no_data'}

        mode_counts = {}
        for entry in self.strategy_history:
            mode = entry['strategy']['recommended_mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        return {
            'total_steps': self.step_count,
            'mode_distribution': mode_counts,
            'most_used_mode': max(mode_counts.items(), key=lambda x: x[1])[0] if mode_counts else None,
            'strategy_changes': len([i for i in range(1, len(self.strategy_history)) 
                                   if self.strategy_history[i]['strategy']['recommended_mode'] != 
                                      self.strategy_history[i-1]['strategy']['recommended_mode']]),
            'uptime': time.time() - (self.strategy_history[0]['timestamp'] if self.strategy_history else time.time())
        }
    
    
    def _compute_trend(self, values) -> str:
        """Compute simple trend (increasing, decreasing, stable) for a list of values"""
        if len(values) < 3:
            return 'insufficient_data'
            
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_current_strategy(self) -> Dict:
        """Get the current strategy recommendation without updating"""
        return self.hhmm.get_strategy_recommendation()
    
    def reset(self):
        """Reset the controller state"""
        self.hhmm = TrueHierarchicalHMMWithBOCPD(use_submodes=False, use_bocpd=True)
        self.performance_buffer.clear()
        self.strategy_history.clear()
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.successful_transitions = 0
        self.total_transitions = 0
        self.last_update_time = time.time()

# Usage example
if __name__ == "__main__":
    # Minimal fake "replay_buffer": list of dicts like your code expects
    rb = []
    hhmm = TrueHierarchicalHMMWithBOCPD(use_submodes=False, use_bocpd=False)

    # Warm-up with exploration-ish evidence
    for step in range(10):
        rb.append({'position': [step, 0], 'place_doubt_step_count': 0})
        beliefs, diag = hhmm.update(rb, external_info_gain=0.6, external_plan_progress=0.1)
        if step % 3 == 0:
            strat = hhmm.get_strategy_recommendation()
            print("[EARLY] mode=", strat['recommended_mode'],
                  "p(TASK)=", diag['mode_probabilities'].get('TASK_SOLVING', 0.0))

    # Ramp plan/task progress to force TASK_SOLVING
    for step in range(10, 25):
        rb.append({'position': [step, 0], 'place_doubt_step_count': 0})
        prog = min(1.0, 0.05*(step-10)+0.1)
        beliefs, diag = hhmm.update(rb, external_info_gain=0.3, external_plan_progress=prog)
        if step % 3 == 0:
            strat = hhmm.get_strategy_recommendation()
            print("[LATE ] prog=", f"{prog:.2f}", "mode=", strat['recommended_mode'],
                  "p(TASK)=", f"{diag['mode_probabilities'].get('TASK_SOLVING', 0.0):.3f}")



if __name__ == "__main__":
        # test_hhmm_4mode.py
    import math
    import types
    import numpy as np
    import importlib
    import sys
    from pathlib import Path

    # Make sure the module is importable
    MOD_NAME = "HierarchicalHMMBOCPD"
    if MOD_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(MOD_NAME, str(Path(__file__).parent / f"{MOD_NAME}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = sys.modules[MOD_NAME]

    TrueH = mod.TrueHierarchicalHMMWithBOCPD

    def make_buffer(n=20, lost=False, loop=False, doubt=0, start=(0.0,0.0), step=(1.0,0.0)):
        """Build a simple replay_buffer the extractor can read."""
        rb = []
        x, y = start
        for t in range(n):
            pos = [x + step[0]*t, y + step[1]*t]
            rb.append({
                "position": pos,
                "place_doubt_step_count": int(doubt),
                # optional extras could go here
            })
        # optionally force "loop" by repeating last few positions
        if loop and n >= 12:
            for k in range(6):
                rb.append({"position": rb[-6]["position"], "place_doubt_step_count": int(doubt)})
        if lost:
            for i in range(5):
                rb[-1 - i]["place_doubt_step_count"] = max(5, int(doubt) + 5)
        return rb

    def test_init_and_shapes():
        h = TrueH(use_submodes=False, use_bocpd=False)
        assert h.states == ['EXPLORE','NAVIGATE','RECOVER','TASK_SOLVING']
        assert h.transition_matrix.shape == (4,4)
        # rows normalized
        rs = h.transition_matrix.sum(axis=1)
        assert np.allclose(rs, 1.0, atol=1e-6)
        # beliefs normalized
        assert math.isclose(float(h.state_beliefs.sum()), 1.0, rel_tol=1e-9)

    def test_emission_params_and_likelihoods_exist():
        h = TrueH(use_submodes=False, use_bocpd=False)
        # emission params present and keyed by modes
        for m in h.states:
            assert m in h.emission_params
            assert isinstance(h.emission_params[m], dict)
        # basic evidence
        ev = {
            "agent_lost": False, "loop_detected": False,
            "stagnation": 0.2, "info_gain": 0.6,
            "exploration_productivity": 0.55, "navigation_progress": 0.1,
            "recovery_effectiveness": 0.2, "task_progress": 0.1
        }
        for m in h.states:
            lik = h.compute_emission_likelihood(ev, m)
            assert lik > 0.0 and np.isfinite(lik)

    def test_update_without_bocpd_exploration_then_task(capsys):
        h = TrueH(use_submodes=False, use_bocpd=False)
        rb = make_buffer(n=12, doubt=0)
        # early: low plan progress -> expect EXPLORE/NAVIGATE mass
        for t in range(6):
            _, diag = h.update(rb[:t+1], external_info_gain=0.6, external_plan_progress=0.1)
        early_task_p = diag['mode_probabilities']['TASK_SOLVING']

        # later: ramp progress -> expect TASK_SOLVING mass goes up
        last_p = 0.0
        for t in range(6, 20):
            prog = min(1.0, 0.06*(t-6)+0.2)
            _, diag = h.update(rb[:t+1], external_info_gain=0.3, external_plan_progress=prog)
            last_p = diag['mode_probabilities']['TASK_SOLVING']

        assert last_p > early_task_p, "TASK_SOLVING should increase as progress rises"

        # sanity on logs sections
        out = capsys.readouterr().out
        assert "[UPD:1] Evidence:" in out
        assert "[UPD:2] task_progress" in out
        assert "[UPD:3] T row sums" in out
        assert "[UPD:5] eLikes:" in out

    def test_gate_function_speeds_task_entry():
        h = TrueH(use_submodes=False, use_bocpd=False)

        # a gate that ramps enter fast and exit low as progress increases
        def gate(self, evidence, tp_smooth):
            enter = 0.02 + 4.0 * (tp_smooth ** 2)   # small -> large
            exit_ = max(0.6 * (1 - tp_smooth), 0.05)
            return enter, exit_
        h.set_task_solving_gate(fn=gate)

        rb = make_buffer(n=25, doubt=0)
        p_task = []
        for t in range(1, 25):
            prog = min(1.0, 0.05*t)
            _, diag = h.update(rb[:t+1], external_info_gain=0.3, external_plan_progress=prog)
            p_task.append(diag['mode_probabilities']['TASK_SOLVING'])

        # should be monotonically trending up overall (allow a tiny wiggle)
        assert p_task[-1] > p_task[3]
        assert max(np.diff(p_task[-8:])) > -0.02  # no strong down-spikes near the end

    def test_strategy_recommendation_round_trip():
        h = TrueH(use_submodes=False, use_bocpd=False)
        rb = make_buffer(n=8, doubt=0)
        for t in range(1, 8):
            _, diag = h.update(rb[:t+1], external_info_gain=0.5, external_plan_progress=0.2)
        strat = h.get_strategy_recommendation()
        assert 'recommended_mode' in strat
        assert strat['recommended_mode'] in h.states
        assert 'mode_confidence' in strat and 0.0 <= strat['mode_confidence'] <= 1.0

    def test_transition_changes_with_gates():
        h = TrueH(use_submodes=False, use_bocpd=False)
        rb = make_buffer(n=5)
        # fixed gates baseline
        _, _ = h.update(rb, external_info_gain=0.3, external_plan_progress=0.1)
        T0 = h.transition_matrix.copy()

        # strong entry to TASK
        h.set_task_solving_gate(enter=10.0, exit=1.0)
        _, _ = h.update(rb + [{"position":[100,0], "place_doubt_step_count":0}], external_info_gain=0.3, external_plan_progress=0.9)
        T1 = h.transition_matrix.copy()

        i = h.state_index['EXPLORE']; j = h.state_index['TASK_SOLVING']
        assert T1[i, j] > T0[i, j], "enter gate should increase EXPLORE->TASK_SOLVING transition weight"

