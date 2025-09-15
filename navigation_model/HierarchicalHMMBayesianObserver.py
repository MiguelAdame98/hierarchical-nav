import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time, math
from scipy import stats
import time
import random
import numpy as np
from nltk import PCFG
import math
import heapq
import torch
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from collections import namedtuple
from nltk.parse.generate import generate
State = namedtuple("State", ["x","y","d"])
DIR_VECS = [(1,0),(0,1),(-1,0),(0,-1)]
ACTIONS   = ["forward","left","right"]

class HierarchicalHMMBayesianObserver:
    """
    A single HMM whose hidden states are (mode, submode) pairs,
    fused with Bayesian Online Changepoint Detection (BOCPD).
    """
    def __init__(self):
        # ── 1) Define your two‐level hierarchy ─────────────────────
        self.mode_submodes = {
            'EXPLORE':        ['ego_allo','ego_allo_lookahead','short_term_memory','astar_directed'],
            'NAVIGATE':       ['distant_node','unvisited_priority','plan_following'],
            'RECOVER':        ['solve_doubt','backtrack_safe'],
            'TASK_SOLVING':   ['goal_directed','systematic_search','task_completion']
        }
        # flatten into atomic states
        self.hierarchical_states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        idx = 0
        for mode,subs in self.mode_submodes.items():
            for sub in subs:
                self.hierarchical_states.append((mode,sub))
                self.state_to_idx[(mode,sub)] = idx
                self.idx_to_state[idx] = (mode,sub)
                idx += 1
        self.n_states = len(self.hierarchical_states)

        # ── 2) BOCPD + evidence as before ────────────────────────
        self.last_poses            = deque(maxlen=100)
        self.replay_buffer         = deque(maxlen=80)
        self.new_nodes_created     = deque(maxlen=50)
        self.plan_progress_history = deque(maxlen=30)
        self.last_significant_progress = time.time()
        self.stagnation_counter    = 0
        self.exploration_attempts  = 0
        self.navigation_attempts   = 0
        self.recovery_attempts     = 0

        self.params = {
            'loop_threshold':20,
            'stagnation_time_threshold':30.0,
            'stagnation_step_threshold':50,
            'max_exploration_cycles':3,
            'max_navigation_cycles':2,
            'max_recovery_cycles':2
        }

        self.max_run_length       = 50
        self.run_length_dist      = np.zeros(self.max_run_length+1)
        self.run_length_dist[0]   = 1.0
        self.hazard_rate          = 1.0/25.0
        self.changepoint_threshold= 0.1

        self.evidence_buffer      = deque(maxlen=100)
        self.evidence_history     = deque(maxlen=self.max_run_length)
        self.mode_history         = deque(maxlen=100)

        self.joint_states = self.hierarchical_states
        self.state_beliefs = np.ones(self.n_states) / self.n_states

        # ── 3) Build your hierarchical HMM parameters ────────────
        self.transition_matrix = self._build_hierarchical_transition_matrix()
        self.emission_params   = self._build_hierarchical_emission_params()

        # beliefs now over the joint states
        self.state_beliefs = np.ones(self.n_states)/self.n_states
        self.modes = list(self.mode_submodes.keys())

    # ──────────────────────────────────────────
    #  Transition matrix builder (TrueHHMM style)
    # ──────────────────────────────────────────
    def _build_hierarchical_transition_matrix(self) -> np.ndarray:
        T = np.zeros((self.n_states, self.n_states))
        for i, (from_m, from_s) in enumerate(self.hierarchical_states):
            row = np.zeros(self.n_states)
            for j, (to_m, to_s) in enumerate(self.hierarchical_states):
                if from_m == to_m:
                    if from_s == to_s:
                        # self‐persistence
                        row[j] = self._get_submode_persistence(from_m, from_s)
                    else:
                        # switch submode within same mode
                        row[j] = self._get_intra_mode_transition(from_m, from_s, to_s)
                else:
                    # jump mode
                    row[j] = self._get_inter_mode_transition(from_m, from_s, to_m, to_s)
            
            # Ensure proper normalization
            row_sum = row.sum()
            if row_sum > 1e-10:
                T[i, :] = row / row_sum
            else:
                # Fallback to uniform if all probabilities are zero
                T[i, :] = 1.0 / self.n_states
                
            # Double-check normalization (should sum to 1.0)
            assert abs(T[i, :].sum() - 1.0) < 1e-6, f"Row {i} doesn't sum to 1.0: {T[i, :].sum()}"
        
        return T


    def get_diagnostics(self) -> Dict:
        """
        Return diagnostic information about the HMM observer state.
        """
        current_mode_probs = self.get_mode_probabilities()
        dominant_mode = max(current_mode_probs.items(), key=lambda x: x[1])
        
        return {
            'current_mode_probabilities': current_mode_probs,
            'dominant_mode': dominant_mode[0],
            'dominant_confidence': dominant_mode[1],
            'current_run_length_dist': self.run_length_dist.copy(),
            'max_run_length_prob': np.max(self.run_length_dist),
            'evidence_buffer_size': len(self.evidence_buffer),
            'evidence_history_size': len(self.evidence_history),
            'buffer_stats': {
                'poses_tracked': len(self.last_poses),
                'replay_buffer_size': len(self.replay_buffer),
                'new_nodes_buffer': len(self.new_nodes_created),
                'plan_progress_history': len(self.plan_progress_history)
            },
            'state_beliefs': self.state_beliefs.copy(),
            'n_states': self.n_states
        }
    def _get_submode_persistence(self, mode:str, sub:str)->float:
        base = 0.7
        # tweak per submode
        if mode=='EXPLORE' and sub=='astar_directed': return 0.8
        if mode=='NAVIGATE' and sub=='plan_following': return 0.85
        if mode=='RECOVER': return 0.5
        if mode=='TASK_SOLVING': return 0.9
        return base
    def _compute_prior_likelihood(self, evidence: Dict) -> float:
        """
        Uniform prior over *all* joint (mode,submode) states.
        """
        # simply average the emission probability under each joint state
        likes = [
            self.compute_emission_likelihood(evidence, state)
            for state in self.hierarchical_states
        ]
        return float(np.mean(likes))
    
    def _compute_joint_weighted_likelihood(self, evidence: Dict, var: str) -> float:
        """
        Soft‐assign fallback: weight the PDF of `var` under each joint state
        by the current joint-state belief.
        """
        total = 0.0
        for idx, state in enumerate(self.hierarchical_states):
            p_state = self.state_beliefs[idx]
            prm = self.emission_params[state].get(var)
            if isinstance(prm, dict):
                total += p_state * stats.norm.pdf(
                    evidence[var],
                    prm['mean'],
                    prm['std']
                )
        return total
    
    def _get_intra_mode_transition(self, mode:str, fsub:str, tsub:str)->float:
        # small base for all submode‐to‐submode
        mapping = {
          'EXPLORE': {
            'ego_allo':{'ego_allo_lookahead':0.15,'short_term_memory':0.1,'astar_directed':0.05},
            'ego_allo_lookahead':{'ego_allo':0.1,'astar_directed':0.1},
            'short_term_memory':{'ego_allo':0.1,'astar_directed':0.15},
            'astar_directed':{'ego_allo':0.1,'short_term_memory':0.05},
          },
          'NAVIGATE': {
            'distant_node':{'unvisited_priority':0.1,'plan_following':0.05},
            'unvisited_priority':{'distant_node':0.1,'plan_following':0.1},
            'plan_following':{'distant_node':0.08,'unvisited_priority':0.05},
          },
          'RECOVER': {'solve_doubt':{'backtrack_safe':0.2}, 'backtrack_safe':{'solve_doubt':0.25}},
          'TASK_SOLVING': {
            'goal_directed':{'systematic_search':0.05,'task_completion':0.02},
            'systematic_search':{'goal_directed':0.08,'task_completion':0.05},
            'task_completion':{'goal_directed':0.03,'systematic_search':0.03},
          }
        }
        return mapping.get(mode,{}).get(fsub,{}).get(tsub, 0.05)

    def _get_inter_mode_transition(self, fm,fs, tm,ts)->float:
        base = {
          'EXPLORE':{'NAVIGATE':0.15,'RECOVER':0.05,'TASK_SOLVING':0.05},
          'NAVIGATE':{'EXPLORE':0.1,'RECOVER':0.15,'TASK_SOLVING':0.05},
          'RECOVER':{'EXPLORE':0.2,'NAVIGATE':0.2,'TASK_SOLVING':0.05},
          'TASK_SOLVING':{'EXPLORE':0.05,'NAVIGATE':0.05,'RECOVER':0.05},
        }[fm].get(tm,0.01)
        # split among target submodes by entry preference
        prefs = {
          'EXPLORE':{'ego_allo':0.4,'ego_allo_lookahead':0.3,'short_term_memory':0.2,'astar_directed':0.1},
          'NAVIGATE':{'distant_node':0.4,'unvisited_priority':0.3,'plan_following':0.3},
          'RECOVER':{'solve_doubt':0.6,'backtrack_safe':0.4},
          'TASK_SOLVING':{'goal_directed':0.5,'systematic_search':0.3,'task_completion':0.2},
        }[tm].get(ts,1.0/len(self.mode_submodes[tm]))
        return base * prefs

    # ──────────────────────────────────────────
    #  Emission parameters builder (per‐state)
    # ──────────────────────────────────────────
    def _build_hierarchical_emission_params(self) -> Dict[Tuple[str,str],Dict]:
        out = {}
        for mode,subs in self.mode_submodes.items():
            for sub in subs:
                out[(mode,sub)] = self._get_emission_params_for_state(mode,sub)
        return out

    def detect_loop_behavior(self, agent_state, env_state) -> bool:
        """
        Simple loop detection based on recent position history.
        Fixed to handle different position formats and edge cases.
        """
        # Extract position with multiple fallback methods
        current_pos = None
        if 'position' in agent_state:
            current_pos = agent_state['position']
        elif 'pos' in agent_state:
            current_pos = agent_state['pos']
        elif hasattr(agent_state, 'position'):
            current_pos = agent_state.position
        else:
            # Try to extract from state if it's a State namedtuple
            if hasattr(agent_state, 'x') and hasattr(agent_state, 'y'):
                current_pos = (agent_state.x, agent_state.y)
            else:
                # Default fallback
                current_pos = (0, 0)
        
        # Ensure position is a tuple
        if not isinstance(current_pos, (tuple, list)):
            current_pos = (float(current_pos), 0.0) if isinstance(current_pos, (int, float)) else (0, 0)
        elif len(current_pos) < 2:
            current_pos = (float(current_pos[0]), 0.0) if len(current_pos) == 1 else (0, 0)
        else:
            current_pos = (float(current_pos[0]), float(current_pos[1]))
        
        self.last_poses.append(current_pos)
        
        if len(self.last_poses) < self.params['loop_threshold']:
            return False
        
        # Check if current position appeared recently
        recent_poses = list(self.last_poses)[-self.params['loop_threshold']:]
        position_counts = {}
        for pos in recent_poses:
            # Use rounded positions to handle floating point precision issues
            rounded_pos = (round(pos[0], 1), round(pos[1], 1))
            position_counts[rounded_pos] = position_counts.get(rounded_pos, 0) + 1
        
        # If any position appears more than 30% of recent steps, consider it a loop
        if not position_counts:
            return False
            
        max_count = max(position_counts.values())
        threshold = len(recent_poses) * 0.3
        return max_count > threshold

    def _get_emission_params_for_state(self, mode:str, sub:str) -> Dict:
        # copy your old mode‐params and tweak
        base = {
            'EXPLORE': {
                'info_gain':{'mean':0.4,'std':0.2},
                'progress': {'mean':0.2,'std':0.15},
                'stagnation':{'mean':0.3,'std':0.2},
                'lost_prob':0.2,
                'loop_prob':0.3
            },
            'NAVIGATE': {
                'info_gain':{'mean':0.1,'std':0.1},
                'progress': {'mean':0.6,'std':0.2},
                'stagnation':{'mean':0.2,'std':0.15},
                'lost_prob':0.15,
                'loop_prob':0.2
            },
            'RECOVER': {
                'info_gain':{'mean':0.05,'std':0.05},
                'progress': {'mean':0.1,'std':0.1},
                'stagnation':{'mean':0.8,'std':0.2},
                'lost_prob':0.9,
                'loop_prob':0.7
            },
            'TASK_SOLVING': {
                'info_gain':{'mean':0.2,'std':0.1},
                'progress': {'mean':0.5,'std':0.2},
                'stagnation':{'mean':0.1,'std':0.1},
                'lost_prob':0.1,
                'loop_prob':0.1
            }
        }[mode].copy()

        # submode tweaks:
        if mode=='EXPLORE':
            if   sub=='ego_allo':           base['info_gain']['mean']+=0.1
            elif sub=='astar_directed':     base['progress']['mean']+=0.2
            elif sub=='short_term_memory':  base['loop_prob']=0.2
        if mode=='NAVIGATE' and sub=='plan_following':
            base['progress']['mean']+=0.15; base['stagnation']['mean']-=0.1
        if mode=='RECOVER' and sub=='solve_doubt':
            base['lost_prob']=0.7
        # TASK_SOLVING left as mode base

        return base

    # ──────────────────────────────────────────
    #  Emission likelihood for a joint state
    # ──────────────────────────────────────────
    def compute_emission_likelihood(self, evidence: Dict, state: Tuple[str,str]) -> float:
        """
        Compute emission likelihood for a joint (mode, submode) state.
        Returns probability (not log-probability) with improved numerical stability.
        """
        if state not in self.emission_params:
            return 1e-10
            
        params = self.emission_params[state]
        log_ll = 0.0
        
        # Continuous variables - use log probabilities and sum
        for var in ['info_gain', 'progress', 'stagnation']:
            if var in evidence and var in params and isinstance(params[var], dict):
                value = float(evidence[var])
                # Clamp value to reasonable range
                value = np.clip(value, -10, 10)
                
                mean = float(params[var]['mean'])
                std = max(float(params[var]['std']), 1e-3)  # Prevent division by zero
                
                # Use scipy.stats for better numerical stability
                try:
                    log_ll += stats.norm.logpdf(value, mean, std)
                except:
                    # Fallback calculation if scipy fails
                    log_ll += -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((value - mean) / std)**2
        
        # Binary variables with better handling
        if 'agent_lost' in evidence and 'lost_prob' in params:
            p_lost = np.clip(float(params['lost_prob']), 1e-10, 1-1e-10)
            if evidence['agent_lost']:
                log_ll += np.log(p_lost)
            else:
                log_ll += np.log(1 - p_lost)
        
        # Handle loop detection if present in evidence
        if 'loop_detected' in evidence and 'loop_prob' in params:
            p_loop = np.clip(float(params['loop_prob']), 1e-10, 1-1e-10)
            if evidence['loop_detected']:
                log_ll += np.log(p_loop)
            else:
                log_ll += np.log(1 - p_loop)
        
        # Convert back to probability with better numerical stability
        log_ll = np.clip(log_ll, -50, 50)  # Prevent extreme values
        return np.exp(log_ll)

    # ──────────────────────────────────────────
    #  BOCPD: exactly as your run‐length specific code
    # ──────────────────────────────────────────
    # ————— BOCPD helpers —————
    def compute_run_length_specific_likelihood(self,
                                               evidence: Dict,
                                               run_length: int) -> float:
        """
        For each candidate run_length, build an empirical norm from the
        last `run_length` observations of each var, and score current evidence.
        Fallback to joint-weighted static if not enough history.
        """
        # 1) changepoint = use uniform prior
        if run_length == 0:
            return self._compute_prior_likelihood(evidence)

        hist = list(self.evidence_history)
        relevant = hist[-run_length:] if len(hist) >= run_length else hist
        if not relevant:
            return self._compute_prior_likelihood(evidence)

        logl = 0.0
        for var in ['info_gain','progress','stagnation',
                    'exploration_productivity',
                    'navigation_progress',
                    'recovery_effectiveness']:
            if var in evidence:
                vals = [e[var] for e in relevant if var in e]
                if len(vals) >= 2:
                    μ = np.mean(vals)
                    σ = max(np.std(vals), 1e-2) * (1 + 1.0/run_length)
                    logl += stats.norm.logpdf(evidence[var], μ, σ)
                else:
                    # not enough history → fallback to static joint mixture
                    logl += np.log(self._compute_joint_weighted_likelihood(evidence, var) + 1e-10)

        # 2) combine with a static fallback on emission likelihood
        #    static_mix = Σ_b(state) · p(evidence | state)
        emis = np.array([
            self.compute_emission_likelihood(evidence, state)
            for state in self.hierarchical_states
        ])
        static_mix = float(np.dot(self.state_beliefs, emis))

        # 3) convex combine
        comb = 0.7 * np.exp(logl) + 0.3 * static_mix
        return max(comb, 1e-10)

    # ──────────────────────────────────────────
    #  HMM forward step (joint states)
    # ──────────────────────────────────────────
    def hmm_forward_step(self, evidence: Dict) -> np.ndarray:
        """
        Standard HMM forward (predict + update) over joint states.
        """
        # emission likelihood under each joint (mode,submode)
        e_likes = np.array([
            self.compute_emission_likelihood(evidence, state)
            for state in self.hierarchical_states
        ])

        # predict
        pred = self.transition_matrix.T @ self.state_beliefs

        # update
        newb = e_likes * pred
        if newb.sum() > 0:
            self.state_beliefs = newb / newb.sum()
        else:
            self.state_beliefs = np.ones(self.n_states) / self.n_states

        return self.state_beliefs
    
    def bocpd_update(self, evidence: Dict) -> Tuple[bool, float]:
        """
        Run‐length recursion using your run_length‐specific likelihoods.
        Fixed array handling and probability computation.
        """
        # Compute likelihoods for each run length
        likelihoods = np.zeros(self.max_run_length + 1)
        for r in range(len(self.run_length_dist)):
            if self.run_length_dist[r] > 1e-10:
                likelihoods[r] = self.compute_run_length_specific_likelihood(evidence, r)
        
        # Growth probabilities (no changepoint)
        growth_probs = np.zeros(self.max_run_length + 1)
        if len(self.run_length_dist) > 1:
            growth_probs[1:] = (self.run_length_dist[:-1] * 
                            (1 - self.hazard_rate) * 
                            likelihoods[:-1])
        
        # Changepoint probability 
        cp_mass = float(np.sum(self.run_length_dist * self.hazard_rate * likelihoods))
        
        # Update run length distribution
        new_dist = np.zeros_like(self.run_length_dist)
        new_dist[0] = cp_mass  # Changepoint mass goes to run length 0
        new_dist[1:] = growth_probs[1:]  # Growth probabilities
        
        # Normalize to ensure it's a proper probability distribution
        total_mass = new_dist.sum()
        if total_mass > 1e-10:
            new_dist /= total_mass
        else:
            # Fallback to uniform if numerical issues
            new_dist = np.ones_like(new_dist) / len(new_dist)
        
        self.run_length_dist = new_dist
        
        # Detect changepoint based on mass at run length 0
        changepoint_detected = cp_mass > self.changepoint_threshold
        
        return changepoint_detected, cp_mass

    # ──────────────────────────────────────────
    #  Full update: BOCPD + forward
    # ──────────────────────────────────────────
    def update(self,evidence:Dict)->Tuple[np.ndarray,bool]:
        # 1) accumulate evidence
        self.evidence_history.append(evidence.copy())
        self.evidence_buffer.append(evidence.copy())
        # 2) BOCPD
        cp_flag, cp_p = self.bocpd_update(evidence)
        if cp_flag:
            # reset joint‐belief “sticky”
            uniform = np.ones(self.n_states)/self.n_states
            self.state_beliefs = 0.5*self.state_beliefs + 0.5*uniform
        # 3) HMM forward
        self.hmm_forward_step(evidence)
        # 4) book-keeping
        self.mode_history.append((time.time(), self.state_beliefs.copy()))
        return self.state_beliefs, cp_flag

    # ──────────────────────────────────────────
    #  Diagnostics: marginalize back to modes
    # ──────────────────────────────────────────
    def get_mode_probabilities(self)->Dict[str,float]:
        marg = defaultdict(float)
        for idx,prob in enumerate(self.state_beliefs):
            mode, _ = self.idx_to_state[idx]
            marg[mode] += prob
        return dict(marg)

    def get_submode_probabilities(self)->Dict[str,Dict[str,float]]:
        mode_probs = self.get_mode_probabilities()
        out = {}
        for mode,subs in self.mode_submodes.items():
            out[mode]={}
            if mode_probs[mode]>1e-10:
                for sub in subs:
                    idx = self.state_to_idx[(mode,sub)]
                    out[mode][sub] = self.state_beliefs[idx]/mode_probs[mode]
            else:
                # uniform fallback
                for sub in subs:
                    out[mode][sub] = 1.0/len(subs)
        return out
# --------------------------------------------------
# PCFG Builder & Controller (unchanged from your skeleton)
# --------------------------------------------------
class EnhancedMixturePCFGBuilder:
    def __init__(self, key, hmm_observer: HierarchicalHMMBayesianObserver):
        self.key          = key
        self.hmm_observer = hmm_observer
        self.memory_graph = key.models_manager.memory_graph
        self.emap         = self.memory_graph.experience_map
        self.submode_beliefs = {
            'EXPLORE': {'ego_allo':0.4,'ego_allo_lookahead':0.3,'short_term_memory':0.2,'astar_directed':0.1},
            'NAVIGATE':{'distant_node':0.4,'unvisited_priority':0.3,'plan_following':0.3},
            'RECOVER': {'solve_doubt':0.6,'backtrack_safe':0.4},
            'TASK_SOLVING':{'goal_directed':0.5,'systematic_search':0.3,'task_completion':0.2}
        }

    def update_submode_beliefs(self,evidence:Dict):
        # your RL-style update from before...
        current = max(self.hmm_observer.get_mode_probabilities().items(),
                      key=lambda x:x[1])[0]
        perf = evidence.get('performance_score',0.5)
        sub = evidence.get('active_submode')
        if sub and sub in self.submode_beliefs[current]:
            lr=0.1; cb=self.submode_beliefs[current][sub]
            if perf>0.6:   self.submode_beliefs[current][sub]=min(0.9,cb+lr*(1-cb))
            elif perf<0.3: self.submode_beliefs[current][sub]=max(0.1,cb-lr*cb)
            # renormalize
            tot=sum(self.submode_beliefs[current].values())
            for k in self.submode_beliefs[current]:
                self.submode_beliefs[current][k]/=tot

    def build_mixture_pcfg(self,use_soft:bool=True)->PCFG:
        mode_probs=self.hmm_observer.get_mode_probabilities()
        if use_soft:
            return self._build_soft_mixture_pcfg(mode_probs)
        else:
            best=max(mode_probs,key=mode_probs.get)
            return self._build_single_mode_pcfg(best)

    def _build_soft_mixture_pcfg(self, mode_probs):
        """
        Build PCFG using current HMM submode beliefs instead of local ones.
        """
        rules = []
        s = sum(mode_probs.values())
        
        # Top-level mode rules
        if s > 0:
            for m, p in mode_probs.items():
                rules.append(f"START -> {m}_ROOT [{p/s:.4f}]")
        else:
            for m in mode_probs: 
                rules.append(f"START -> {m}_ROOT [0.25]")
        
        # Get current submode beliefs from HMM observer
        current_submode_beliefs = self.hmm_observer.get_submode_probabilities()
        
        # Build rules using HMM submode beliefs
        rules += self._build_explore_rules(current_submode_beliefs.get('EXPLORE', {}))
        rules += self._build_navigate_rules(current_submode_beliefs.get('NAVIGATE', {}))
        rules += self._build_recover_rules(current_submode_beliefs.get('RECOVER', {}))
        rules += self._build_task_solving_rules(current_submode_beliefs.get('TASK_SOLVING', {}))
        
        return PCFG.fromstring("\n".join(rules))

    def _build_explore_rules(self, submode_probs=None):
        """Fixed to use consistent terminal names and proper probability formatting."""
        if submode_probs is None:
            submode_probs = self.submode_beliefs['EXPLORE']
        
        p = submode_probs
        s = sum(p.values()) if p else 1.0
        
        if s > 0:
            # Fixed: Use consistent naming that matches the terminal definitions below
            mapping = {
                'ego_allo': 'EGOALLO',
                'ego_allo_lookahead': 'EGOALLOLOOKAHEAD', 
                'short_term_memory': 'SHORTTERMMEMORY',
                'astar_directed': 'ASTARDIRECTED'
            }
            r = [f"EXPLORE_ROOT -> EXPLORE_{mapping.get(k, k.upper())} [{(v/s):.4f}]" for k, v in p.items()]
        else:
            # Fallback to uniform distribution
            r = [
                "EXPLORE_ROOT -> EXPLORE_EGOALLO [0.25]",
                "EXPLORE_ROOT -> EXPLORE_EGOALLOLOOKAHEAD [0.25]", 
                "EXPLORE_ROOT -> EXPLORE_SHORTTERMMEMORY [0.25]",
                "EXPLORE_ROOT -> EXPLORE_ASTARDIRECTED [0.25]"
            ]
        
        r += [
            "EXPLORE_EGOALLO -> 'forward' [0.6] | 'left' [0.2] | 'right' [0.2]",
            "EXPLORE_EGOALLOLOOKAHEAD -> 'forward' 'forward' [0.4] | 'left' 'forward' [0.3] | 'right' 'forward' [0.3]", 
            "EXPLORE_SHORTTERMMEMORY -> 'scan' [0.3] | 'forward' [0.4] | 'backtrack' [0.3]",
            "EXPLORE_ASTARDIRECTED -> 'plan_to_frontier' [1.0]"
        ]
        return r

    def _build_navigate_rules(self, submode_probs=None):
        """Fixed to use consistent terminal names."""
        if submode_probs is None:
            submode_probs = self.submode_beliefs['NAVIGATE']
        
        p = submode_probs
        s = sum(p.values()) if p else 1.0
        
        if s > 0:
            mapping = {
                'distant_node': 'DISTANTNODE',
                'unvisited_priority': 'UNVISITEDPRIORITY',
                'plan_following': 'PLANFOLLOWING'
            }
            r = [f"NAVIGATE_ROOT -> NAVIGATE_{mapping.get(k, k.upper())} [{(v/s):.4f}]" for k, v in p.items()]
        else:
            r = [
                "NAVIGATE_ROOT -> NAVIGATE_DISTANTNODE [0.33]",
                "NAVIGATE_ROOT -> NAVIGATE_UNVISITEDPRIORITY [0.33]",
                "NAVIGATE_ROOT -> NAVIGATE_PLANFOLLOWING [0.34]"
            ]
        
        r += [
            "NAVIGATE_DISTANTNODE -> 'goto_distant' [1.0]",
            "NAVIGATE_UNVISITEDPRIORITY -> 'goto_unvisited' [1.0]", 
            "NAVIGATE_PLANFOLLOWING -> 'follow_plan' [0.8] | 'replan' [0.2]"
        ]
        return r

    def _build_recover_rules(self, submode_probs=None):
        """Fixed to use consistent terminal names."""
        if submode_probs is None:
            submode_probs = self.submode_beliefs['RECOVER']
        
        p = submode_probs  
        s = sum(p.values()) if p else 1.0
        
        if s > 0:
            mapping = {
                'solve_doubt': 'SOLVEDOUBT',
                'backtrack_safe': 'BACKTRACKSAFE'
            }
            r = [f"RECOVER_ROOT -> RECOVER_{mapping.get(k, k.upper())} [{(v/s):.4f}]" for k, v in p.items()]
        else:
            r = [
                "RECOVER_ROOT -> RECOVER_SOLVEDOUBT [0.5]",
                "RECOVER_ROOT -> RECOVER_BACKTRACKSAFE [0.5]"
            ]
        
        r += [
            "RECOVER_SOLVEDOUBT -> 'scan' [0.4] | 'relocalize' [0.6]",
            "RECOVER_BACKTRACKSAFE -> 'backtrack' [0.7] | 'return_to_known' [0.3]"
        ]
        return r

    def _build_task_solving_rules(self, submode_probs=None):
        """Fixed to use consistent terminal names."""
        if submode_probs is None:
            submode_probs = self.submode_beliefs['TASK_SOLVING']
        
        p = submode_probs
        s = sum(p.values()) if p else 1.0
        
        if s > 0:
            mapping = {
                'goal_directed': 'GOALDIRECTED',
                'systematic_search': 'SYSTEMATICSEARCH', 
                'task_completion': 'TASKCOMPLETION'
            }
            r = [f"TASK_SOLVING_ROOT -> TASK_{mapping.get(k, k.upper())} [{(v/s):.4f}]" for k, v in p.items()]
        else:
            r = [
                "TASK_SOLVING_ROOT -> TASK_GOALDIRECTED [0.33]",
                "TASK_SOLVING_ROOT -> TASK_SYSTEMATICSEARCH [0.33]",
                "TASK_SOLVING_ROOT -> TASK_TASKCOMPLETION [0.34]"
            ]
        
        r += [
            "TASK_GOALDIRECTED -> 'execute_task_plan' [0.8] | 'refine_task_plan' [0.2]",
            "TASK_SYSTEMATICSEARCH -> 'systematic_exploration' [0.6] | 'check_all_rooms' [0.4]", 
            "TASK_TASKCOMPLETION -> 'complete_objective' [0.9] | 'verify_completion' [0.1]"
        ]
        return r

class EnhancedHybridBayesianController:
    def __init__(self, key, buffer_size:int=50):
        self.key           = key
        self.hmm_observer  = HierarchicalHMMBayesianObserver()
        self.pcfg_builder  = EnhancedMixturePCFGBuilder(key,self.hmm_observer)
        self.performance_buffer = deque(maxlen=buffer_size)
        self.last_evidence     = {}
        self.use_soft_mixture  = True
        self.adaptation_enabled = True

    def extract_enhanced_evidence(self, agent_state, env_state, perf):
        """
        Extract evidence dictionary that matches what HMM expects.
        Improved error handling and type checking.
        """
        # Safely extract values with defaults
        def safe_get(obj, key, default=0.0):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        # Calculate derived metrics with bounds checking
        info_gain = float(safe_get(perf, 'info_gain', 0.0))
        if info_gain == 0.0:
            # Fallback: derive info_gain from new nodes created
            new_nodes = float(safe_get(env_state, 'new_nodes', 0))
            info_gain = min(1.0, max(0.0, new_nodes / 10.0))
        
        progress = float(safe_get(perf, 'plan_progress', 0.0))
        if progress == 0.0:
            has_goal = bool(safe_get(agent_state, 'has_navigation_goal', False))
            if has_goal:
                # Derive progress from goal proximity or path completion
                steps_since = float(safe_get(agent_state, 'steps_since_progress', 0))
                progress = max(0.0, 1.0 - min(1.0, steps_since / 20.0))
        
        # Calculate stagnation score
        doubt_count = float(safe_get(agent_state, 'place_doubt_step_count', 0))
        steps_since_progress = float(safe_get(agent_state, 'steps_since_progress', 0))
        stagnation = min(1.0, max(0.0, (doubt_count + steps_since_progress) / 30.0))
        
        # Compute additional metrics expected by BOCPD
        exploration_productivity = max(0.0, info_gain * (1.0 - stagnation))
        navigation_progress = progress if bool(safe_get(agent_state, 'has_navigation_goal', False)) else 0.0
        recovery_effectiveness = max(0.0, 1.0 - stagnation) if doubt_count > 5 else 0.0
        
        # Get reward with fallback
        reward = float(safe_get(perf, 'reward', 0.5))
        if reward == 0.5:  # Default fallback, try other keys
            reward = float(safe_get(perf, 'performance_score', 0.5))
        
        e = {
            'timestamp': time.time(),
            'performance_score': np.clip(reward, 0.0, 1.0),
            'active_submode': safe_get(agent_state, 'active_submode', None),
            
            # Core HMM evidence variables (all clipped to [0,1])
            'info_gain': np.clip(info_gain, 0.0, 1.0),
            'progress': np.clip(progress, 0.0, 1.0), 
            'stagnation': np.clip(stagnation, 0.0, 1.0),
            'agent_lost': bool(doubt_count > 6),
            
            # Additional BOCPD variables
            'exploration_productivity': np.clip(exploration_productivity, 0.0, 1.0),
            'navigation_progress': np.clip(navigation_progress, 0.0, 1.0),
            'recovery_effectiveness': np.clip(recovery_effectiveness, 0.0, 1.0),
            
            # Raw state information (keep as is for debugging)
            'place_doubt_step_count': int(doubt_count),
            'new_nodes': int(safe_get(env_state, 'new_nodes', 0)),
            'nodes_created': int(safe_get(env_state, 'nodes_created_total', 0)),
            'plan_progress': np.clip(progress, 0.0, 1.0),
            'has_navigation_goal': bool(safe_get(agent_state, 'has_navigation_goal', False)),
            'path_blocked': bool(safe_get(env_state, 'path_blocked', False)),
            'task_defined': bool(safe_get(agent_state, 'task_defined', False)),
            'exploration_completeness': np.clip(float(safe_get(env_state, 'exploration_completeness', 0.0)), 0.0, 1.0),
            'steps_since_progress': int(steps_since_progress)
        }
        return e

    def update(self, agent_state, env_state, perf):
        """
        Main update method for the controller.
        """
        # Add loop detection to evidence
        loop_detected = self.hmm_observer.detect_loop_behavior(agent_state, env_state)
        
        # Extract evidence with loop detection included
        evidence = self.extract_enhanced_evidence(agent_state, env_state, perf)
        evidence['loop_detected'] = loop_detected
        
        self.last_evidence = evidence
        self.performance_buffer.append(perf.get('reward', 0.0))

        # 1) HMM+BOCPD update
        beliefs, cp = self.hmm_observer.update(evidence)
        
        # 2) Submode adaptation
        if self.adaptation_enabled:
            self.pcfg_builder.update_submode_beliefs(evidence)
        
        # 3) Build PCFG
        pcfg = self.pcfg_builder.build_mixture_pcfg(self.use_soft_mixture)
        
        # 4) Compile diagnostics
        diag = {
            'mode_beliefs': self.hmm_observer.get_mode_probabilities(),
            'submode_beliefs': self.hmm_observer.get_submode_probabilities(),
            'changepoint': cp,
            'changepoint_probability': float(np.sum(self.hmm_observer.run_length_dist * self.hmm_observer.hazard_rate)),
            'last_evidence': evidence,
            'loop_detected': loop_detected,
            **self.hmm_observer.get_diagnostics()
        }
        
        return pcfg, diag

    def get_current_strategy(self)->Tuple[str,float]:
        mp = self.hmm_observer.get_mode_probabilities()
        best,conf = max(mp.items(),key=lambda x:x[1])
        return best,conf

    def toggle_mixture_mode(self,use_soft:Optional[bool]=None):
        if use_soft is not None: self.use_soft_mixture = use_soft
        else:                   self.use_soft_mixture = not self.use_soft_mixture
        print(f"[Controller] {'soft' if self.use_soft_mixture else 'hard'} mixture")

    def print_status(self):
        best,conf = self.get_current_strategy()
        stats = self.hmm_observer.get_diagnostics()['buffer_stats']
        print(f"Dominant: {best} ({conf:.2f}); poses={stats['poses_tracked']}, replay={stats['replay_buffer_size']}")
