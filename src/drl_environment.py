import gymnasium as gym
from gymnasium import spaces
import numpy as np

from xfoil_wrapper import run_xfoil
# Import the new validation function
from airfoil_geometry import generate_coords_from_cpts, calculate_area, is_geometry_valid
from reward_functions import calculate_reward

class AirfoilOptEnv(gym.Env):
    # In drl_environment.py

    def __init__(self, baseline_cpts_file='baseline_control_points.npz'):
        super(AirfoilOptEnv, self).__init__()

        baseline_data = np.load(baseline_cpts_file)
        self.baseline_x_cpts = baseline_data['x_cpts']
        self.baseline_y_cpts_upper = baseline_data['y_cpts_upper']
        self.baseline_y_cpts_lower = baseline_data['y_cpts_lower']
        
        self.baseline_area = 0.13689111 

        self.num_cpts = len(self.baseline_x_cpts)
        self.state = np.concatenate([self.baseline_y_cpts_upper, self.baseline_y_cpts_lower])

        num_actions = 2 * (self.num_cpts - 2)
        
        # --- CHANGE: Reduce the action limit significantly ---
        action_limit = 0.0001 # Was 0.005, now much smaller
        # --- END CHANGE ---
        
        self.action_space = spaces.Box(low=-action_limit, high=action_limit, shape=(num_actions,), dtype=np.float32)

        num_observations = 2 * self.num_cpts
        obs_limit = 0.5
        self.observation_space = spaces.Box(low=-obs_limit, high=obs_limit, shape=(num_observations,), dtype=np.float32)

        print("DRL Environment Initialized with Area Constraint.")
        print(f"Target Baseline Area: >= {self.baseline_area * 0.98:.6f} (98% of {self.baseline_area:.6f})")
        print(f"Action limit set to ±{action_limit}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.concatenate([self.baseline_y_cpts_upper, self.baseline_y_cpts_lower])
        return self.state.astype(np.float32), {}

    def step(self, action):
        current_y_upper = self.state[:self.num_cpts].copy()
        current_y_lower = self.state[self.num_cpts:].copy()
        action_upper = action[:self.num_cpts - 2]
        action_lower = action[self.num_cpts - 2:]
        current_y_upper[1:-1] += action_upper
        current_y_lower[1:-1] += action_lower
        self.state = np.concatenate([current_y_upper, current_y_lower])

        x_coords, y_coords = generate_coords_from_cpts(
            self.baseline_x_cpts, current_y_upper, current_y_lower
        )
        
        # --- NEW: GEOMETRY VALIDITY CHECK ---
        if not is_geometry_valid(x_coords, y_coords):
            # If the geometry is invalid, don't run XFOIL.
            # Terminate the episode with a large penalty.
            reward = -10.0
            terminated = True
            truncated = False
            info = {'error': 'Invalid geometry'}
            return self.state.astype(np.float32), reward, terminated, truncated, info
        # --- END OF NEW CHECK ---

        new_area = calculate_area(x_coords, y_coords)

        polar = run_xfoil(
            x_coords, y_coords, airfoil_name="drl_airfoil", reynolds=500000, 
            alpha_start=0, alpha_end=20, alpha_step=1.0
        )

        reward = calculate_reward(polar, new_area, self.baseline_area, area_tolerance=0.98)
        
        terminated = False
        if reward <= -4.0:
            terminated = True
        
        truncated = False
        info = {'polar': polar, 'area': new_area}

        return self.state.astype(np.float32), reward, terminated, truncated, info