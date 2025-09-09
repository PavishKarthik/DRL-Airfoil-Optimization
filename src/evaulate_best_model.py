# File: evaluate_best_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from drl_environment import AirfoilOptEnv
from airfoil_geometry import generate_coords_from_cpts, calculate_area
from xfoil_wrapper import run_xfoil
from reward_functions import calculate_reward

print("--- Evaluating the Best Trained Model ---")

# --- CONFIGURATION ---
MODEL_DIR = "ppo_airfoil_models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
BASELINE_NPZ_PATH = "baseline_control_points.npz"
# --- ADDITION: This MUST match the value in your environment ---
ACTION_LIMIT = 0.001 
# --- END ADDITION ---

# --- 1. LOAD THE BASELINE GEOMETRY & CALCULATE ITS AREA ---
baseline_data = np.load(BASELINE_NPZ_PATH)
baseline_x_cpts = baseline_data['x_cpts']
baseline_y_cpts_upper = baseline_data['y_cpts_upper']
baseline_y_cpts_lower = baseline_data['y_cpts_lower']
baseline_x_coords, baseline_y_coords = generate_coords_from_cpts(
    baseline_x_cpts, baseline_y_cpts_upper, baseline_y_cpts_lower
)
baseline_area = calculate_area(baseline_x_coords, baseline_y_coords)
print(f"Loaded baseline geometry. Baseline Area = {baseline_area:.6f}")

# --- 2. LOAD THE TRAINED DRL AGENT ---
if os.path.exists(MODEL_PATH):
    print(f"Loading best model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# --- 3. GET THE OPTIMIZED SHAPE FROM THE AGENT ---
env = AirfoilOptEnv(baseline_cpts_file=BASELINE_NPZ_PATH)
obs, _ = env.reset()
# Get the agent's raw, NORMALIZED action
raw_action, _states = model.predict(obs, deterministic=True)

# --- FIX: MANUALLY SCALE THE NORMALIZED ACTION TO THE PHYSICAL RANGE ---
scaled_action = raw_action * ACTION_LIMIT
# --- END FIX ---

# --- 4. CREATE THE OPTIMIZED AIRFOIL GEOMETRY ---
# Apply the SCALED optimal perturbation to the baseline control points
optimized_y_cpts_upper = baseline_y_cpts_upper.copy()
optimized_y_cpts_lower = baseline_y_cpts_lower.copy()

num_movable_cpts = len(baseline_x_cpts) - 2
optimized_y_cpts_upper[1:-1] += scaled_action[:num_movable_cpts]
optimized_y_cpts_lower[1:-1] += scaled_action[num_movable_cpts:]

optimized_x_coords, optimized_y_coords = generate_coords_from_cpts(
    baseline_x_cpts, optimized_y_cpts_upper, optimized_y_cpts_lower
)
print("\nOptimized airfoil shape has been generated.")

# --- 5. EVALUATE THE OPTIMIZED AIRFOIL'S PERFORMANCE ---
print("Running XFOIL analysis on the optimized geometry...")
optimized_area = calculate_area(optimized_x_coords, optimized_y_coords)
optimized_polar = run_xfoil(
    optimized_x_coords, optimized_y_coords,
    airfoil_name="drl_optimized_airfoil",
    reynolds=500000, 
    alpha_start=0, 
    alpha_end=20, 
    alpha_step=1.0
)

if optimized_polar is not None:
    optimized_reward = calculate_reward(optimized_polar, optimized_area, baseline_area)
    print("\n--- Optimized Performance Results ---")
    print(f"Optimized Airfoil Area: {optimized_area:.6f}")
    print(f"Optimized Reward Score: {optimized_reward:.4f}")
    print("\nOptimized Polar Data:")
    print(optimized_polar.to_string())
else:
    print("\n--- XFOIL analysis failed for the optimized airfoil. ---")

# --- 6. PLOT THE COMPARISON ---
plt.figure(figsize=(16, 8))
plt.plot(baseline_x_coords, baseline_y_coords, 'k--', label='Original 8M Airfoil (Baseline)')
plt.plot(optimized_x_coords, optimized_y_coords, 'b-', linewidth=2, label='DRL Optimized Airfoil')
plt.plot(baseline_x_cpts, optimized_y_cpts_upper, 'o', color='gray', label='Original Upper CPs')
plt.plot(baseline_x_cpts, baseline_y_cpts_lower, 'o', color='gray')
plt.plot(baseline_x_cpts, optimized_y_cpts_upper, 'ro-', label='Optimized Upper CPs')
plt.plot(baseline_x_cpts, optimized_y_cpts_lower, 'go-', label='Optimized Lower CPs')
plt.title('Comparison of Original vs. DRL Optimized Airfoil', fontsize=16)
plt.xlabel('x/c', fontsize=12)
plt.ylabel('y/c', fontsize=12)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

env.close()