# File: reward_functions.py
import numpy as np
import pandas as pd

def calculate_reward(
    polar_df, 
    is_valid_geometry,
    new_area, 
    baseline_area,
    new_max_thickness,
    baseline_max_thickness,
    area_tolerance=0.95
):
    """
    Calculates a multi-objective reward score based on aerodynamic performance
    and several critical engineering constraints.

    Args:
        polar_df (pd.DataFrame or None): The XFOIL simulation results.
        is_valid_geometry (bool): Flag indicating if the geometry was self-intersecting.
        new_area (float): The cross-sectional area of the new airfoil.
        baseline_area (float): The area of the baseline airfoil.
        new_max_thickness (float): The max thickness of the new airfoil.
        baseline_max_thickness (float): The max thickness of the baseline airfoil.
        area_tolerance (float): The minimum allowable area as a fraction of the baseline.

    Returns:
        float: The calculated total reward.
    """
    
    # --- 1. HANDLE CATASTROPHIC FAILURES ---
    # This is the primary "hard stop" penalty. If the geometry is invalid or the
    # simulation fails, the episode must terminate with a large, unambiguous penalty.
    if not is_valid_geometry or polar_df is None or polar_df.empty:
        return -50.0

    # --- 2. CALCULATE ENGINEERING CONSTRAINT PENALTIES ---
    
    # Area Constraint Penalty (Ensures Structural Integrity)
    area_penalty = 0.0
    min_allowable_area = baseline_area * area_tolerance
    if new_area < min_allowable_area:
        area_penalty = -25.0

    # Max Thickness Constraint Penalty (Ensures Aerodynamic Realism)
    thickness_penalty = 0.0
    max_allowable_thickness = baseline_max_thickness * 1.10
    if new_max_thickness > max_allowable_thickness:
        thickness_penalty = -25.0

    # Pitching Moment Constraint Penalty (Ensures Stability)
    avg_cm = polar_df['CM'].mean()
    # A ramped penalty is applied if the moment is more negative than -0.08,
    # encouraging the agent to find more stable (less nose-down) designs.
    cm_penalty = 100 * min(0, avg_cm + 0.08)

    # --- 3. CALCULATE AERODYNAMIC PERFORMANCE REWARD ---
    
    # Pre-Stall Lift (The Primary Project Objective)
    pre_stall_angles = polar_df[(polar_df['alpha'] >= 4) & (polar_df['alpha'] <= 10)]
    avg_pre_stall_cl = pre_stall_angles['CL'].mean() if not pre_stall_angles.empty else -1.0

    # Post-Stall Lift (Preserving the Bio-Inspired Strength)
    post_stall_angles = polar_df[(polar_df['alpha'] >= 15) & (polar_df['alpha'] <= 20)]
    avg_post_stall_cl = post_stall_angles['CL'].mean() if not post_stall_angles.empty else -1.0

    # Drag Penalty (Encourages Efficiency)
    avg_cd = polar_df['CD'].mean()
    drag_penalty = -0.1 * avg_cd

    # Combine the weighted aerodynamic objectives
    aero_reward = (2.0 * avg_pre_stall_cl) + (1.0 * avg_post_stall_cl) + drag_penalty

    # --- 4. ADD BEHAVIORAL INCENTIVE ---
    
    # "Alive" Bonus (Encourages Agent Survival and Robustness)
    # This bonus is critical for encouraging the agent to maintain valid geometries
    # and explore for longer periods, directly addressing the decreasing episode length.
    alive_bonus = 0.1

    # --- 5. COMBINE ALL METRICS INTO THE FINAL REWARD ---
    
    total_reward = (
        aero_reward + 
        area_penalty + 
        thickness_penalty + 
        cm_penalty + 
        alive_bonus
    )

    # Final sanity check for NaN values, which can crash the training process.
    if np.isnan(total_reward):
        return -50.0 # Return the catastrophic failure penalty

    return total_reward