import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time
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
        self.hazard_rate = 1.0 / 25.0
        self.changepoint_threshold = 0.1
        
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
    
    def _get_intra_mode_transition(self, mode: str, from_sub: str, to_sub: str) -> float:
        """Probability of switching submodes within the same mode"""
        base_prob = 0.05  # Low base probability
        
        if mode == 'EXPLORE':
            # Define natural submode progressions within exploration
            transitions = {
                'ego_allo': {'ego_allo_lookahead': 0.15, 'short_term_memory': 0.1, 'astar_directed': 0.05},
                'ego_allo_lookahead': {'ego_allo': 0.1, 'astar_directed': 0.1},
                'short_term_memory': {'ego_allo': 0.1, 'astar_directed': 0.15},
                'astar_directed': {'ego_allo': 0.1, 'short_term_memory': 0.05}
            }
        elif mode == 'NAVIGATE':
            transitions = {
                'distant_node': {'unvisited_priority': 0.1, 'plan_following': 0.05},
                'unvisited_priority': {'distant_node': 0.1, 'plan_following': 0.1},
                'plan_following': {'distant_node': 0.08, 'unvisited_priority': 0.05}
            }
        elif mode == 'RECOVER':
            transitions = {
                'solve_doubt': {'backtrack_safe': 0.2},
                'backtrack_safe': {'solve_doubt': 0.25}
            }
        elif mode == 'TASK_SOLVING':
            transitions = {
                'goal_directed': {'systematic_search': 0.05, 'task_completion': 0.02},
                'systematic_search': {'goal_directed': 0.08, 'task_completion': 0.05},
                'task_completion': {'goal_directed': 0.03, 'systematic_search': 0.03}
            }
        else:
            transitions = {}
            
        return transitions.get(from_sub, {}).get(to_sub, base_prob)
    
    def _get_inter_mode_transition(self, from_mode: str, from_sub: str, to_mode: str, to_sub: str) -> float:
        """Probability of transitioning between modes"""
        
        # Base inter-mode transition probabilities
        mode_transitions = {
            'EXPLORE': {'NAVIGATE': 0.15, 'RECOVER': 0.05, 'TASK_SOLVING': 0.05},
            'NAVIGATE': {'EXPLORE': 0.1, 'RECOVER': 0.15, 'TASK_SOLVING': 0.05},
            'RECOVER': {'EXPLORE': 0.2, 'NAVIGATE': 0.2, 'TASK_SOLVING': 0.05},
            'TASK_SOLVING': {'EXPLORE': 0.05, 'NAVIGATE': 0.05, 'RECOVER': 0.05}
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
        Updated emission parameters - removed reward-based parameters, 
        made evidence mode-specific
        """
        
        if mode == 'EXPLORE':
            return {
                'info_gain': {'mean': 0.4, 'std': 0.2},
                'stagnation': {'mean': 0.3, 'std': 0.2},
                'exploration_productivity': {'mean': 0.4, 'std': 0.2},
                'lost_prob': 0.2,
                'loop_prob': 0.3,
                'new_node_prob': 0.3 if submode == 'astar_directed' else 0.2
            }
        elif mode == 'NAVIGATE':
            return {
                'navigation_progress': {'mean': 0.6, 'std': 0.2},
                'stagnation': {'mean': 0.2, 'std': 0.15},
                'lost_prob': 0.15,
                'loop_prob': 0.2
            }
        elif mode == 'RECOVER':
            return {
                'recovery_effectiveness': {'mean': 0.3, 'std': 0.2},
                'stagnation': {'mean': 0.8, 'std': 0.2},
                'lost_prob': 0.9,
                'loop_prob': 0.7
            }
        elif mode == 'TASK_SOLVING':
            return {
                'task_progress': {'mean': 0.5, 'std': 0.2},
                'stagnation': {'mean': 0.1, 'std': 0.1},
                'lost_prob': 0.1,
                'loop_prob': 0.1
            }
    
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
    def extract_evidence_from_replay_buffer(self, replay_buffer, external_info_gain=None, external_plan_progress=None) -> Dict:
        """
        Extract mode-specific evidence from replay buffer and external sources
        """
        evidence = {}
        
        # Get current most likely mode to determine relevant evidence
        current_mode, _ = self.get_most_likely_state()
        
        # UNIVERSAL EVIDENCE (used by all modes)
        evidence['agent_lost'] = self.detect_agent_lost(replay_buffer)
        evidence['loop_detected'] = self.detect_loop_behavior(replay_buffer)
        evidence['movement_score'] = self.calculate_pose_movement(replay_buffer)
        evidence['stagnation'] = 1.0 - evidence['movement_score']  # Inverse of movement
        
        # MODE-SPECIFIC EVIDENCE
        if current_mode == 'EXPLORE':
            # Only use info_gain for exploration modes
            if external_info_gain is not None:
                evidence['info_gain'] = float(external_info_gain)
            else:
                evidence['info_gain'] = 0.5  # Default moderate info gain
            
            # Exploration-specific metrics
            evidence['new_node_created'] = self.detect_new_node_created(replay_buffer)
            evidence['exploration_productivity'] = evidence['info_gain'] * (1.0 - evidence['stagnation'])
            
        elif current_mode == 'NAVIGATE':
            # Only use plan_progress for navigation modes
            if external_plan_progress is not None:
                evidence['plan_progress'] = float(external_plan_progress)
                evidence['navigation_progress'] = evidence['plan_progress']
            else:
                evidence['navigation_progress'] = 0.5  # Default moderate progress
            
        elif current_mode == 'RECOVER':
            # Recovery-specific evidence
            evidence['recovery_effectiveness'] = self.calculate_recovery_effectiveness(replay_buffer)
            
        elif current_mode == 'TASK_SOLVING':
            # Task-solving specific evidence
            if external_plan_progress is not None:
                evidence['task_progress'] = float(external_plan_progress)
            
        return evidence
    def detect_agent_lost(self, replay_buffer) -> bool:
        """
        Detect if agent seems lost based on replay buffer
        """
        if len(replay_buffer) < 5:
            return False
            
        # Check for indicators of being lost (implement based on your criteria)
        # For example, checking for repeated failed actions or high uncertainty
        recent_entries = list(replay_buffer)[-5:]
        
        # Placeholder logic - implement based on your specific indicators
        return False
    def calculate_recovery_effectiveness(self, replay_buffer) -> float:
        """
        Calculate how effective recovery attempts have been
        """
        if len(replay_buffer) < 10:
            return 0.5
            
        # Calculate based on position changes during recovery
        movement = self.calculate_pose_movement(replay_buffer)
        return movement  # Simple proxy for recovery effectiveness
    # ===== BOCPD Methods =====
    
    def detect_loop_behavior(self, replay_buffer) -> bool:
        """
        Detect loops using external replay buffer positions
        """
        if len(replay_buffer) < self.params['loop_threshold']:
            return False
            
        # Extract positions from last N entries in replay buffer
        recent_positions = []
        for entry in list(replay_buffer)[-self.params['loop_threshold']:]:
            if 'real_pose' in entry:
                pos = entry['real_pose']
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    recent_positions.append((float(pos[0]), float(pos[1])))
        
        if len(recent_positions) < self.params['loop_threshold']:
            return False
            
        # Check for repeated positions
        position_counts = {}
        for pos in recent_positions:
            rounded_pos = (round(pos[0], 1), round(pos[1], 1))
            position_counts[rounded_pos] = position_counts.get(rounded_pos, 0) + 1
        
        if not position_counts:
            return False
            
        max_count = max(position_counts.values())
        threshold = len(recent_positions) * 0.3
        return max_count > threshold
    
    def detect_new_node_created(self, replay_buffer) -> bool:
        """
        Check if a new node was created in recent steps
        """
        if len(replay_buffer) < 2:
            return False
            
        # Check last few entries for new node creation
        recent_entries = list(replay_buffer)[-5:]  # Check last 5 steps
        
        for entry in recent_entries:
            if 'node_id' in entry:
                # If we have a way to detect new nodes, implement here
                # For now, simple check if node_id changed recently
                pass
        
        return False  # Implement based on your node tracking logic
    def calculate_pose_movement(self, replay_buffer) -> float:
        """
        Calculate movement/stagnation from replay buffer poses
        """
        if len(replay_buffer) < 10:
            return 0.5  # Default moderate movement
            
        recent_poses = []
        for entry in list(replay_buffer)[-10:]:
            if 'real_pose' in entry:
                pose = entry['real_pose']
                if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                    recent_poses.append((float(pose[0]), float(pose[1])))
        
        if len(recent_poses) < 2:
            return 0.5
            
        # Calculate total distance moved
        total_distance = 0.0
        for i in range(1, len(recent_poses)):
            dx = recent_poses[i][0] - recent_poses[i-1][0]
            dy = recent_poses[i][1] - recent_poses[i-1][1]
            total_distance += (dx**2 + dy**2)**0.5
        
        # Normalize to 0-1 range (adjust scale as needed)
        movement_score = min(total_distance / 10.0, 1.0)  # Adjust divisor based on your scale
        return movement_score
    
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
    
    def compute_run_length_specific_likelihood(self, evidence: Dict, run_length: int) -> float:
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
    
    def bocpd_update(self, evidence: Dict) -> Tuple[bool, float]:
        """
        Run‐length recursion using run_length‐specific likelihoods.
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
    
    def update(self, replay_buffer, external_info_gain=None, external_plan_progress=None) -> Tuple[np.ndarray, Dict]:
        """
        Updated method that uses external replay buffer and specific evidence
        
        Args:
            replay_buffer: External deque with agent state history
            external_info_gain: Optional info_gain value from image comparison
            external_plan_progress: Optional plan progress value during navigation
        """
        
        # Extract evidence from replay buffer and external sources
        evidence = self.extract_evidence_from_replay_buffer(
            replay_buffer, external_info_gain, external_plan_progress
        )
        
        # Store evidence for BOCPD
        self.evidence_buffer.append(evidence)
        self.evidence_history.append(evidence.copy())

        # Run BOCPD update first
        changepoint_detected, cp_probability = self.bocpd_update(evidence)

        # If changepoint detected, partially reset beliefs
        if changepoint_detected:
            uniform = np.ones(self.n_states) / self.n_states
            self.state_beliefs = 0.5 * self.state_beliefs + 0.5 * uniform
            print(f"[BOCPD] Changepoint detected with probability: {cp_probability:.3f}")

        # Run HMM forward step
        self.hmm_forward_step(evidence)

        # Store history
        self.state_history.append((time.time(), self.state_beliefs.copy()))
        self.mode_history.append((time.time(), self.state_beliefs.copy()))

        # Compute marginal probabilities
        mode_probs = self.get_mode_probabilities()
        submode_probs = self.get_submode_probabilities()

        diagnostics = {
            'mode_probabilities': mode_probs,
            'submode_probabilities': submode_probs,
            'most_likely_state': self.get_most_likely_state(),
            'state_entropy': self.compute_state_entropy(),
            'mode_entropy': self.compute_mode_entropy(),
            'changepoint_detected': changepoint_detected,
            'changepoint_probability': cp_probability,
            'run_length_dist': self.run_length_dist.copy(),
            'max_run_length_prob': np.max(self.run_length_dist),
            'evidence_buffer_size': len(self.evidence_buffer),
            'evidence_history_size': len(self.evidence_history),
            'evidence_used': evidence  # Add evidence for debugging
        }

        return self.state_beliefs, diagnostics
    
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
        self.key = key
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
    
    
    def _enhance_diagnostics(self, base_diagnostics, strategy, evidence) -> Dict:
        """Add controller-specific diagnostics to base HMM diagnostics"""
        enhanced = base_diagnostics.copy()
        
        # Add performance history
        enhanced['performance_history'] = {
            'cumulative_reward': self.cumulative_reward,
            'step_count': self.step_count,
            'average_reward': self.cumulative_reward / max(self.step_count, 1),
            'recent_performance': list(self.performance_buffer)[-10:] if self.performance_buffer else []
        }
        
        # Add strategy stability metrics
        if len(self.strategy_history) > 1:
            recent_strategies = [s['strategy']['recommended_mode'] for s in list(self.strategy_history)[-10:]]
            strategy_changes = sum(1 for i in range(1, len(recent_strategies)) 
                                 if recent_strategies[i] != recent_strategies[i-1])
            enhanced['strategy_stability'] = {
                'recent_changes': strategy_changes,
                'stability_ratio': 1.0 - (strategy_changes / max(len(recent_strategies) - 1, 1))
            }
        
        # Add evidence quality metrics
        enhanced['evidence_quality'] = {
            'info_gain_trend': self._compute_trend([e['evidence']['info_gain'] 
                                                  for e in list(self.strategy_history)[-10:]]),
            'progress_trend': self._compute_trend([e['evidence']['progress'] 
                                                 for e in list(self.strategy_history)[-10:]]),
            'stagnation_trend': self._compute_trend([e['evidence']['stagnation'] 
                                                   for e in list(self.strategy_history)[-10:]])
        }
        
        # Add timing information
        enhanced['timing'] = {
            'last_update': self.last_update_time,
            'update_frequency': len(self.strategy_history) / max(time.time() - self.strategy_history[0]['timestamp'], 1) 
                              if self.strategy_history else 0
        }
        
        return enhanced
    
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
