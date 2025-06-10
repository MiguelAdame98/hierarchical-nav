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
    def __init__(self):
        # Define the hierarchy
        self.mode_submodes = {
            'EXPLORE': ['ego_allo', 'ego_allo_lookahead', 'short_term_memory', 'astar_directed'],
            'NAVIGATE': ['distant_node', 'unvisited_priority', 'plan_following'],
            'RECOVER': ['solve_doubt', 'backtrack_safe'],
            'TASK_SOLVING': ['goal_directed', 'systematic_search', 'task_completion']
        }
        
        # Create flat state space of (mode, submode) pairs
        self.hierarchical_states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        idx = 0
        for mode, submodes in self.mode_submodes.items():
            for submode in submodes:
                state = (mode, submode)
                self.hierarchical_states.append(state)
                self.state_to_idx[state] = idx
                self.idx_to_state[idx] = state
                idx += 1
        
        self.n_states = len(self.hierarchical_states)
        print(f"[HHMM+BOCPD] Created {self.n_states} hierarchical states")
        
        # Initialize transition matrix and emission parameters
        self.transition_matrix = self._build_hierarchical_transition_matrix()
        self.emission_params = self._build_hierarchical_emission_params()
        
        # State beliefs (now over (mode, submode) pairs)
        self.state_beliefs = np.ones(self.n_states) / self.n_states
        
        # ===== BOCPD Integration =====
        # BOCPD parameters
        self.max_run_length = 50
        self.run_length_dist = np.zeros(self.max_run_length + 1)
        self.run_length_dist[0] = 1.0
        self.hazard_rate = 1.0 / 15.0
        self.changepoint_threshold = 0.1
        self.step_counter =0
        
        # Evidence and history tracking for BOCPD
        self.evidence_buffer = deque(maxlen=100)
        self.evidence_history = deque(maxlen=self.max_run_length)
        self.mode_history = deque(maxlen=100)
        
        # Enhanced tracking from BOCPD version
        self.new_nodes_created = deque(maxlen=50)
        self.plan_progress_history = deque(maxlen=30)


        self.last_significant_progress = time.time()
        self.stagnation_counter = 0
        self.exploration_attempts = 0
        self.navigation_attempts = 0
        self.recovery_attempts = 0

        self.prev_evidence = None 
        
        # Parameters for behavior detection
        self.params = {
            'loop_threshold': 20,
            'stagnation_time_threshold': 30.0,
            'stagnation_step_threshold': 50,
            'max_exploration_cycles': 3,
            'max_navigation_cycles': 2,
            'max_recovery_cycles': 2
        }
        
        # History tracking
        self.state_history = deque(maxlen=100)
        
    def _build_hierarchical_transition_matrix(self) -> np.ndarray:
        """
        Build transition matrix with hierarchical structure:
        - High intra-mode transitions
        - Lower inter-mode transitions  
        - Submode transitions within modes follow different patterns
        """
        T = np.zeros((self.n_states, self.n_states))
        
        for i, (from_mode, from_submode) in enumerate(self.hierarchical_states):
            # Get all possible transitions from this state
            row_probs = np.zeros(self.n_states)
            
            for j, (to_mode, to_submode) in enumerate(self.hierarchical_states):
                if from_mode == to_mode:
                    # Intra-mode transitions (submode switching)
                    if from_submode == to_submode:
                        # Self-transition (stay in same submode)
                        row_probs[j] = self._get_submode_persistence(from_mode, from_submode)
                    else:
                        # Switch to different submode within same mode
                        row_probs[j] = self._get_intra_mode_transition(from_mode, from_submode, to_submode)
                else:
                    # Inter-mode transitions (mode switching)
                    row_probs[j] = self._get_inter_mode_transition(from_mode, from_submode, to_mode, to_submode)
            
            # Normalize row
            row_sum = row_probs.sum()
            if row_sum > 1e-10:
                T[i, :] = row_probs / row_sum
            else:
                T[i, :] = 1.0 / self.n_states  # Uniform fallback
                
            # Ensure proper normalization
            assert abs(T[i, :].sum() - 1.0) < 1e-6, f"Row {i} doesn't sum to 1.0: {T[i, :].sum()}"
                
        return T
    
    def _get_submode_persistence(self, mode: str, submode: str) -> float:
        """How likely to stay in the same (mode, submode)"""
        base_persistence = 0.7  # Generally sticky
        
        # Mode-specific persistence patterns
        if mode == 'EXPLORE':
            if submode == 'ego_allo':
                return 0.6  # Moderate persistence, allows switching
            elif submode == 'astar_directed':
                return 0.8  # High persistence for directed exploration
        elif mode == 'NAVIGATE':
            if submode == 'plan_following':
                return 0.85  # Very persistent when following a plan
        elif mode == 'RECOVER':
            return 0.5  # Low persistence, recovery should be quick
        elif mode == 'TASK_SOLVING':
            return 0.9  # High persistence for task completion
            
        return base_persistence
    
    def _get_intra_mode_transition(self, mode: str,
                               from_sub: str, to_sub: str) -> float:
        """
        Directed sub-mode transitions.
        Values are *unnormalised* weights; the caller
        will re-scale the entire row so it sums to 1.
        """

        # ---------- EXPLORE ---------------------------------------------------
        if mode == 'EXPLORE':
            transitions = {
                'ego_allo': {
                    'ego_allo_lookahead': 0.25,   # Natural progression
                    'short_term_memory': 0.05,
                    'astar_directed':    0.02
                },
                'ego_allo_lookahead': {
                    'astar_directed':    0.20,
                    'ego_allo':          0.08,
                    'short_term_memory': 0.02
                },
                'short_term_memory': {
                    'ego_allo':          0.15,
                    'astar_directed':    0.10,
                    'ego_allo_lookahead':0.05
                },
                'astar_directed': {
                    'ego_allo':          0.05,
                    'short_term_memory': 0.02,
                    'ego_allo_lookahead':0.08
                }
            }

        # ---------- NAVIGATE ---------------------------------------------------
        elif mode == 'NAVIGATE':
            # “distant_node → plan_following → unvisited_priority → distant_node”
            transitions = {
                'distant_node': {
                    'plan_following':     0.25,
                    'unvisited_priority': 0.05
                },
                'plan_following': {
                    'unvisited_priority': 0.20,
                    'distant_node':       0.08
                },
                'unvisited_priority': {
                    'distant_node':       0.15,
                    'plan_following':     0.05
                }
            }

        # ---------- RECOVER ----------------------------------------------------
        elif mode == 'RECOVER':
            # Simple toggle: solve_doubt ⇄ backtrack_safe (fast)
            transitions = {
                'solve_doubt':   {'backtrack_safe': 0.35},
                'backtrack_safe':{'solve_doubt':   0.40}
            }

        # ---------- TASK_SOLVING ----------------------------------------------
        elif mode == 'TASK_SOLVING':
            # funnel towards 'task_completion'
            transitions = {
                'goal_directed': {
                    'systematic_search': 0.05,
                    'task_completion':   0.20
                },
                'systematic_search': {
                    'goal_directed':     0.08,
                    'task_completion':   0.25
                },
                'task_completion': {
                    'goal_directed':     0.03,
                    'systematic_search': 0.03
                }
            }

        # ---------- default ----------------------------------------------------
        else:
            transitions = {}

        # default tiny weight for unspecified off-diagonal hops
        return transitions.get(from_sub, {}).get(to_sub, 0.01)
    
    def _get_inter_mode_transition(self, from_mode: str, from_sub: str, to_mode: str, to_sub: str) -> float:
        """Probability of transitioning between modes"""
        
        # Base inter-mode transition probabilities
        mode_transitions = {
            'EXPLORE': {'NAVIGATE': 0.15, 'RECOVER': 0.09, 'TASK_SOLVING': 0.01},
            'NAVIGATE': {'EXPLORE': 0.1, 'RECOVER': 0.19, 'TASK_SOLVING': 0.01},
            'RECOVER': {'EXPLORE': 0.6, 'NAVIGATE': 0.2, 'TASK_SOLVING': 0.01},
            'TASK_SOLVING': {'EXPLORE': 0.05, 'NAVIGATE': 0.01, 'RECOVER': 0.09}
        }
        
        base_mode_prob = mode_transitions.get(from_mode, {}).get(to_mode, 0.01)
        
        # Distribute mode transition probability among submodes
        n_target_submodes = len(self.mode_submodes[to_mode])
        
        # Some submodes are more "natural" entry points
        entry_preferences = {
            'EXPLORE': {'ego_allo': 0.4, 'ego_allo_lookahead': 0.3, 'short_term_memory': 0.2, 'astar_directed': 0.1},
            'NAVIGATE': {'distant_node': 0.4, 'unvisited_priority': 0.3, 'plan_following': 0.3},
            'RECOVER': {'solve_doubt': 0.6, 'backtrack_safe': 0.4},
            'TASK_SOLVING': {'goal_directed': 0.5, 'systematic_search': 0.3, 'task_completion': 0.2}
        }
        
        submode_entry_prob = entry_preferences.get(to_mode, {}).get(to_sub, 1.0/n_target_submodes)
        
        return base_mode_prob * submode_entry_prob
    
    def _build_hierarchical_emission_params(self) -> Dict:
        """Build emission parameters for each (mode, submode) pair"""
        params = {}
        
        for mode, submodes in self.mode_submodes.items():
            for submode in submodes:
                state = (mode, submode)
                params[state] = self._get_emission_params_for_state(mode, submode)
                
        return params
    
    def _get_emission_params_for_state(self, mode: str, submode: str) -> Dict:
        """
        Emission parameters with strong differentiation *within* every mode.
        Tune means/σ to your empirical logs.
        """

        # ─────────────────────────────  EXPLORE  ──────────────────────────
        if mode == 'EXPLORE':
            if submode == 'ego_allo':
                return {
                    'info_gain':               {'mean': 0.30, 'std': 0.15},
                    'stagnation':              {'mean': 0.40, 'std': 0.20},
                    'exploration_productivity':{'mean': 0.25, 'std': 0.15},
                    'lost_prob': 0.25, 'loop_prob': 0.40, 'new_node_prob': 0.15
                }
            if submode == 'ego_allo_lookahead':
                return {
                    'info_gain':               {'mean': 0.50, 'std': 0.10},
                    'stagnation':              {'mean': 0.25, 'std': 0.15},
                    'exploration_productivity':{'mean': 0.45, 'std': 0.10},
                    'lost_prob': 0.15, 'loop_prob': 0.20, 'new_node_prob': 0.25
                }
            if submode == 'short_term_memory':
                return {
                    'info_gain':               {'mean': 0.35, 'std': 0.25},
                    'stagnation':              {'mean': 0.35, 'std': 0.30},
                    'exploration_productivity':{'mean': 0.30, 'std': 0.25},
                    'lost_prob': 0.30, 'loop_prob': 0.35, 'new_node_prob': 0.20
                }
            if submode == 'astar_directed':
                return {
                    'info_gain':               {'mean': 0.60, 'std': 0.08},
                    'stagnation':              {'mean': 0.15, 'std': 0.10},
                    'exploration_productivity':{'mean': 0.55, 'std': 0.08},
                    'lost_prob': 0.10, 'loop_prob': 0.15, 'new_node_prob': 0.40
                }

        # ─────────────────────────────  NAVIGATE  ─────────────────────────
        if mode == 'NAVIGATE':
            if submode == 'distant_node':
                return {
                    'navigation_progress': {'mean': 0.30, 'std': 0.20},
                    'stagnation':          {'mean': 0.35, 'std': 0.15},
                    'lost_prob': 0.20, 'loop_prob': 0.25
                }
            if submode == 'unvisited_priority':
                return {
                    'navigation_progress': {'mean': 0.45, 'std': 0.18},
                    'stagnation':          {'mean': 0.25, 'std': 0.12},
                    'lost_prob': 0.18, 'loop_prob': 0.18
                }
            if submode == 'plan_following':
                return {
                    'navigation_progress': {'mean': 0.70, 'std': 0.12},
                    'stagnation':          {'mean': 0.15, 'std': 0.10},
                    'lost_prob': 0.10, 'loop_prob': 0.12
                }

        # ─────────────────────────────  RECOVER  ──────────────────────────
        if mode == 'RECOVER':
            if submode == 'solve_doubt':
                return {
                    'recovery_effectiveness': {'mean': 0.35, 'std': 0.20},
                    'stagnation':             {'mean': 0.75, 'std': 0.15},
                    'lost_prob': 0.95, 'loop_prob': 0.70
                }
            if submode == 'backtrack_safe':
                return {
                    'recovery_effectiveness': {'mean': 0.55, 'std': 0.18},
                    'stagnation':             {'mean': 0.65, 'std': 0.18},
                    'lost_prob': 0.85, 'loop_prob': 0.60
                }

        # ───────────────────────────  TASK_SOLVING  ───────────────────────
        if mode == 'TASK_SOLVING':
            if submode == 'goal_directed':
                return {
                    'task_progress': {'mean': 0.60, 'std': 0.15},
                    'stagnation':    {'mean': 0.12, 'std': 0.08},
                    'lost_prob': 0.12, 'loop_prob': 0.12
                }
            if submode == 'systematic_search':
                return {
                    'task_progress': {'mean': 0.40, 'std': 0.20},
                    'stagnation':    {'mean': 0.18, 'std': 0.10},
                    'lost_prob': 0.18, 'loop_prob': 0.15
                }
            if submode == 'task_completion':
                return {
                    'task_progress': {'mean': 0.80, 'std': 0.10},
                    'stagnation':    {'mean': 0.08, 'std': 0.05},
                    'lost_prob': 0.08, 'loop_prob': 0.08
                }

        # ───────────────────────────  fallback  ───────────────────────────
        # identical to old behaviour if nothing above matched
        return super()._get_emission_params_for_state(mode, submode)
    
    def compute_emission_likelihood(self, evidence: Dict, state: Tuple[str, str]) -> float:
        """
        Updated emission likelihood - only uses evidence relevant to the mode
        """
        if state not in self.emission_params:
            return 1e-10

        params = self.emission_params[state]
        mode, submode = state
        log_ll = 0.0

        # Universal evidence (all modes use these)
        if 'agent_lost' in evidence and 'lost_prob' in params:
            p_lost = np.clip(float(params['lost_prob']), 1e-10, 1-1e-10)
            if evidence['agent_lost']:
                log_ll += np.log(p_lost)
            else:
                log_ll += np.log(1 - p_lost)

        if 'loop_detected' in evidence and 'loop_prob' in params:
            p_loop = np.clip(float(params['loop_prob']), 1e-10, 1-1e-10)
            if evidence['loop_detected']:
                log_ll += np.log(p_loop)
            else:
                log_ll += np.log(1 - p_loop)

        if 'stagnation' in evidence and 'stagnation' in params:
            value = float(evidence['stagnation'])
            mean = float(params['stagnation']['mean'])
            std = max(float(params['stagnation']['std']), 1e-3)
            log_ll += stats.norm.logpdf(value, mean, std)

        # Mode-specific evidence
        if mode == 'EXPLORE':
            if 'info_gain' in evidence and 'info_gain' in params:
                value = float(evidence['info_gain'])
                mean = float(params['info_gain']['mean'])
                std = max(float(params['info_gain']['std']), 1e-3)
                log_ll += stats.norm.logpdf(value, mean, std)
                
            if 'exploration_productivity' in evidence and 'exploration_productivity' in params:
                value = float(evidence['exploration_productivity'])
                mean = float(params['exploration_productivity']['mean'])
                std = max(float(params['exploration_productivity']['std']), 1e-3)
                log_ll += stats.norm.logpdf(value, mean, std)

        elif mode == 'NAVIGATE':
            if 'navigation_progress' in evidence and 'navigation_progress' in params:
                value = float(evidence['navigation_progress'])
                mean = float(params['navigation_progress']['mean'])
                std = max(float(params['navigation_progress']['std']), 1e-3)
                log_ll += stats.norm.logpdf(value, mean, std)

        elif mode == 'RECOVER':
            if 'recovery_effectiveness' in evidence and 'recovery_effectiveness' in params:
                value = float(evidence['recovery_effectiveness'])
                mean = float(params['recovery_effectiveness']['mean'])
                std = max(float(params['recovery_effectiveness']['std']), 1e-3)
                log_ll += stats.norm.logpdf(value, mean, std)

        elif mode == 'TASK_SOLVING':
            if 'task_progress' in evidence and 'task_progress' in params:
                value = float(evidence['task_progress'])
                mean = float(params['task_progress']['mean'])
                std = max(float(params['task_progress']['std']), 1e-3)
                log_ll += stats.norm.logpdf(value, mean, std)

        # Convert back to probability with numerical stability
        log_ll = np.clip(log_ll, -50, 50)
        return np.exp(log_ll)
    def extract_evidence_from_replay_buffer(self, replay_buffer, external_info_gain=None, external_plan_progress=None):
        """
        Centralized evidence extraction that calculates all metrics once and reuses them.
        This replaces multiple separate calculations with a single comprehensive analysis.
        """
        evidence = {}
        
        if len(replay_buffer) < 5:
            # Return default values for insufficient data
            return {
                'agent_lost': False,
                'loop_detected': False,
                'movement_score': 0.5,
                'stagnation': 0.5,
                'info_gain': float(external_info_gain) if external_info_gain is not None else 0.3,
                'exploration_productivity': 0.3,
                'navigation_progress': float(external_plan_progress) if external_plan_progress is not None else 0.3,
                'plan_progress': float(external_plan_progress) if external_plan_progress is not None else 0.3,
                'task_progress': float(external_plan_progress) if external_plan_progress is not None else 0.3,
                'recovery_effectiveness': 0.5,
                'new_node_created': False
            }
        
        # Convert replay buffer to list for easier manipulation
        buffer_list = list(replay_buffer)
        recent_entries = buffer_list[-10:]  # Last 10 entries for most calculations
        
        # ===== CENTRALIZED METRIC CALCULATIONS =====
        
        # 1. Extract poses and positions
        poses = []
        positions = []
        
        for entry in recent_entries:
            if 'real_pose' in entry:
                p = entry['real_pose']

                # Accept dict
                if isinstance(p, dict) and {'x','y'}.issubset(p):
                    poses.append( (p['x'], p['y'], p.get('theta', 0.0)) )
                    positions.append( (float(p['x']), float(p['y'])) )

                # Accept list / tuple
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    poses.append(p)
                    positions.append( (float(p[0]), float(p[1])) )

                # Accept numpy / tensor
                elif isinstance(p, np.ndarray) and p.size >= 2:
                    p = p.astype(float)
                    poses.append(p.tolist())
                    positions.append( (p[0], p[1]) )

                elif torch.is_tensor(p) and p.numel() >= 2:
                    p = p.detach().cpu().float().tolist()
                    poses.append(p)
                    positions.append( (p[0], p[1]) )
# 2. Extract doubt counts (primary recovery/lost indicator)
        doubt_counts = []
        current_doubt_count = 0
        for entry in recent_entries:
            if 'place_doubt_step_count' in entry:
                doubt_count = entry['place_doubt_step_count']
                doubt_counts.append(doubt_count)
                current_doubt_count = doubt_count  # Keep updating to get latest
        
        # 3. Calculate movement metrics
        movement_metrics = self._calculate_movement_metrics(poses, positions)
        
        # 4. Calculate position analysis
        position_metrics = self._calculate_position_metrics(positions, recent_entries)
        
        # 5. Calculate doubt trend analysis
        doubt_metrics = self._calculate_doubt_metrics(doubt_counts)
        
        # ===== BUILD EVIDENCE USING CENTRALIZED METRICS =====
        
        # Universal evidence
        evidence['agent_lost'] = self._determine_agent_lost(
            current_doubt_count, movement_metrics, position_metrics
        )
        evidence['loop_detected'] = self._determine_loop_detected(
            position_metrics, buffer_list
        )
        evidence['movement_score'] = movement_metrics['movement_score']
        evidence['stagnation'] = 1.0 - movement_metrics['movement_score']
        
        # Mode-specific evidence with external inputs
        if external_info_gain is not None:
            evidence['info_gain'] = float(external_info_gain)
            evidence['exploration_productivity'] = (
                evidence['info_gain'] * (1.0 - evidence['stagnation']) * 
                movement_metrics['exploration_factor']
            )
        else:
            evidence['info_gain'] = movement_metrics['exploration_factor'] * 0.6
            evidence['exploration_productivity'] = evidence['info_gain'] * (1.0 - evidence['stagnation'])
        
        if external_plan_progress is not None:
            progress_value = float(external_plan_progress)
            evidence['navigation_progress'] = progress_value
            evidence['plan_progress'] = progress_value
            evidence['task_progress'] = progress_value
        else:
            # Estimate progress from movement and doubt patterns
            estimated_progress = self._estimate_progress_from_movement(
                movement_metrics, doubt_metrics, position_metrics
            )
            evidence['navigation_progress'] = estimated_progress
            evidence['plan_progress'] = estimated_progress
            evidence['task_progress'] = estimated_progress
        
        # Recovery effectiveness using centralized metrics
        evidence['recovery_effectiveness'] = self._calculate_recovery_effectiveness_centralized(
            doubt_metrics, movement_metrics, position_metrics
        )
        
        # New node detection
        evidence['new_node_created'] = self._detect_new_node_created_centralized(recent_entries)
        
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

    def _calculate_recovery_effectiveness_centralized(self, doubt_metrics, movement_metrics, position_metrics):
        """Calculate recovery effectiveness using centralized metrics"""
        if doubt_metrics['current_doubt'] == 0:
            return 0.9  # Excellent recovery
        
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

    # ───────────────────────────────────────────────────────────────────
    #  5.  HMM forward step
    # ───────────────────────────────────────────────────────────────────
    def hmm_forward_step(self, evidence: Dict) -> np.ndarray:
        e_likes = np.array([
            self.compute_emission_likelihood(evidence, s)
            for s in self.hierarchical_states
        ])

        pred = self.transition_matrix.T @ self.state_beliefs
        newb = e_likes * pred
        if newb.sum() > 0:
            self.state_beliefs = newb / newb.sum()
        else:
            self.state_beliefs[:] = 1.0 / self.n_states

        print(f"[DBG-HMM]    eLikes={self._vec2str(e_likes)}  "
            f"pred={self._vec2str(pred)}  "
            f"newBelief={self._vec2str(self.state_beliefs)}")
        return self.state_beliefs

    # ───────────────────────────────────────────────────────────────────
    #  6.  GLOBAL UPDATE ENTRY POINT
    # ───────────────────────────────────────────────────────────────────
    def update(self, replay_buffer, external_info_gain=None,
            external_plan_progress=None) -> Tuple[np.ndarray, Dict]:

        # step counter ---------------------------------------------------
        self.step_counter += 1
        print(f"\n================ STEP {self.step_counter} =================")

        # evidence extraction -------------------------------------------
        evidence = self.extract_evidence_from_replay_buffer(
            replay_buffer, external_info_gain, external_plan_progress
        )
        print(f"[DBG-UPD]   extracted evidence: {evidence}")

        # store / history ----------------------------------------------
        self.evidence_buffer.append(evidence)
        self.evidence_history.append(evidence.copy())
        evidence2 = self._add_deltas(evidence)
        # --- BOCPD -----------------------------------------------------
        changep, cp_prob = self.bocpd_update(evidence2)
        if changep:
            print(f"[DBG-BOCPD] >>> CHG-PT triggered, prob={cp_prob:.3f}")

            # 2.1  Determine how hard to reset
            #       • cp_prob near 1.0  → hard reset (w = 0.9)
            #       • cp_prob near thr  → gentle reset (w = 0.3)
            thr = self.changepoint_threshold
            w = 0.3 + 0.6 * (cp_prob - thr) / max(1.0 - thr, 1e-6)   # clamp [0.3,0.9]
            w = np.clip(w, 0.3, 0.9)

            # 2.2  Build "anti-dominant" proposal
            mode_probs = self.get_mode_probabilities()
            dominant_mode = max(mode_probs.items(), key=lambda x: x[1])[0]

            proposal = np.ones(self.n_states) / self.n_states
            for i, (mode, _) in enumerate(self.hierarchical_states):
                if mode == dominant_mode:
                    proposal[i] *= 0.3          # suppress
                else:
                    proposal[i] *= 1.2          # boost
            proposal /= proposal.sum()

            # 2.3  Blend with previous belief vector
            self.state_beliefs = (1.0 - w) * self.state_beliefs + w * proposal

            # 2.4  Reset the run-length distribution to "just changed"
            self.run_length_dist[:] = 0.0
            self.run_length_dist[0] = 1.0

        # --- HMM forward update ---------------------------------------
        self.hmm_forward_step(evidence)

        # --- bookkeeping ----------------------------------------------
        self.state_history.append((time.time(), self.state_beliefs.copy()))
        self.mode_history.append((time.time(), self.state_beliefs.copy()))

        mode_probs = self.get_mode_probabilities()
        print(f"[DBG-UPD]   modeProbs   = {mode_probs}")
        print(f"[DBG-UPD]   entropy(S)  = {self.compute_state_entropy():.3f}")

        diagnostics = {
            'mode_probabilities': mode_probs,
            'submode_probabilities': self.get_submode_probabilities(),
            'most_likely_state': self.get_most_likely_state(),
            'state_entropy': self.compute_state_entropy(),
            'mode_entropy': self.compute_mode_entropy(),
            'changepoint_detected': changep,
            'changepoint_probability': cp_prob,
            'run_length_dist': self.run_length_dist.copy(),
            'max_run_length_prob': np.max(self.run_length_dist),
            'evidence_buffer_size': len(self.evidence_buffer),
            'evidence_history_size': len(self.evidence_history),
            'evidence_used': evidence
        }
        return self.state_beliefs, diagnostics
    
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
    def get_mode_probabilities(self) -> Dict[str, float]:
        """Compute marginal probabilities over modes"""
        mode_probs = defaultdict(float)
        
        for i, (mode, submode) in enumerate(self.hierarchical_states):
            mode_probs[mode] += self.state_beliefs[i]
            
        return dict(mode_probs)
    
    def get_submode_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Compute conditional probabilities over submodes given modes"""
        mode_probs = self.get_mode_probabilities()
        submode_probs = {}
        
        for mode in self.mode_submodes.keys():
            submode_probs[mode] = {}
            if mode_probs[mode] > 1e-10:  # Avoid division by zero
                for submode in self.mode_submodes[mode]:
                    state = (mode, submode)
                    idx = self.state_to_idx[state]
                    submode_probs[mode][submode] = self.state_beliefs[idx] / mode_probs[mode]
            else:
                # Uniform distribution if mode probability is too low
                n_submodes = len(self.mode_submodes[mode])
                for submode in self.mode_submodes[mode]:
                    submode_probs[mode][submode] = 1.0 / n_submodes
                    
        return submode_probs
    
    def get_most_likely_state(self) -> Tuple[str, str]:
        """Get the most likely (mode, submode) state"""
        idx = np.argmax(self.state_beliefs)
        return self.idx_to_state[idx]
    
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
        """Get concrete strategy recommendation with BOCPD insights"""
        most_likely_mode, most_likely_submode = self.get_most_likely_state()
        mode_probs = self.get_mode_probabilities()
        submode_probs = self.get_submode_probabilities()
        
        return {
            'recommended_mode': most_likely_mode,
            'recommended_submode': most_likely_submode,
            'mode_confidence': mode_probs[most_likely_mode],
            'submode_confidence': submode_probs[most_likely_mode][most_likely_submode],
            'mode_probabilities': mode_probs,
            'submode_probabilities': submode_probs[most_likely_mode],
            'uncertainty': self.compute_state_entropy(),
            'changepoint_mass': self.run_length_dist[0],
            'most_likely_run_length': np.argmax(self.run_length_dist)
        }
    

    # Complete the get_diagnostics method (the end was cut off)
    def get_diagnostics(self) -> Dict:
        """
        Return comprehensive diagnostic information about the HHMM+BOCPD state.
        """
        current_mode_probs = self.get_mode_probabilities()
        dominant_mode = max(current_mode_probs.items(), key=lambda x: x[1])
        
        current_submode_probs = self.get_submode_probabilities()
        most_likely_state = self.get_most_likely_state()
        
        return {
            'state_beliefs': self.state_beliefs.copy(),
            'mode_probabilities': current_mode_probs,
            'submode_probabilities': current_submode_probs,
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
        self.hhmm = TrueHierarchicalHMMWithBOCPD()
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
        self.hhmm = TrueHierarchicalHMMWithBOCPD()
        self.performance_buffer.clear()
        self.strategy_history.clear()
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.successful_transitions = 0
        self.total_transitions = 0
        self.last_update_time = time.time()

# Usage example
if __name__ == "__main__":
    # Initialize controller
    controller = HierarchicalBayesianController()
    
    # Example usage in an agent loop
    for step in range(100):
        # Mock agent and environment state
        agent_state = {
            'position': np.random.randn(2),
            'place_doubt_step_count': np.random.randint(0, 10)
        }
        
        env_state = {
            'visited_nodes': set(range(np.random.randint(1, 20))),
            'total_nodes': 25
        }
        
        perf = {
            'info_gain': np.random.uniform(0, 1),
            'plan_progress': np.random.uniform(0, 1),
            'exploration_efficiency': np.random.uniform(0, 1),
            'reward': np.random.uniform(-0.1, 0.1)
        }
        
        # Update controller
        strategy, diagnostics = controller.update(agent_state, env_state, perf)
        
        # Print strategy every 20 steps
        if step % 20 == 0:
            print(f"Step {step}: Mode={strategy['recommended_mode']}, "
                  f"Submode={strategy['recommended_submode']}, "
                  f"Confidence={strategy['mode_confidence']:.3f}")
    
    # Print final summary
    print("\nFinal Summary:")
    summary = controller.get_summary_stats()
    for key, value in summary.items():
        print(f"{key}: {value}")
