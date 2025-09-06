# File: reward_functions.py
import numpy as np
import pandas as pd

def calculate_reward(polar_df, new_area, baseline_area, area_tolerance=0.98):
    """
    Calculates a reward score, including a penalty for violating the 98% area constraint.
    """
    if polar_df is None or polar_df.empty:
        return -10.0 # Large penalty for simulation failure

    # --- Area Constraint Penalty ---
    area_penalty = 0.0
    # Check if the new area is less than 98% of the baseline area
    if new_area < (baseline_area * area_tolerance):
        # Apply a large, fixed penalty. This signal is unambiguous to the agent.
        area_penalty = -5.0

    # --- Aerodynamic Performance Metrics ---
    pre_stall_angles = polar_df[(polar_df['alpha'] >= 4) & (polar_df['alpha'] <= 10)]
    avg_pre_stall_cl = pre_stall_angles['CL'].mean() if not pre_stall_angles.empty else -1.0

    post_stall_angles = polar_df[(polar_df['alpha'] >= 15) & (polar_df['alpha'] <= 20)]
    avg_post_stall_cl = post_stall_angles['CL'].mean() if not post_stall_angles.empty else -1.0

    avg_cd = polar_df['CD'].mean()
    drag_penalty = -0.1 * avg_cd

    # --- Combine metrics into a final reward ---
    aero_reward = (2.0 * avg_pre_stall_cl) + (1.0 * avg_post_stall_cl) + drag_penalty
    
    total_reward = aero_reward + area_penalty

    if np.isnan(total_reward):
        return -10.0

    return total_reward