import numpy as np

# --- MODIFICATION: Import the calculate_area function ---
from airfoil_geometry import generate_coords_from_cpts, calculate_area
from xfoil_wrapper import run_xfoil
from reward_functions import calculate_reward

def evaluate_baseline_performance(baseline_cpts_file='baseline_control_points.npz'):
    """
    Loads the baseline airfoil, runs an XFOIL analysis, and calculates its reward score
    using the updated, constraint-aware reward function.
    """
    print("--- Evaluating Baseline 8M Airfoil Performance ---")

    # 1. Load the baseline control points
    try:
        baseline_data = np.load(baseline_cpts_file)
        x_cpts = baseline_data['x_cpts']
        y_cpts_upper = baseline_data['y_cpts_upper']
        y_cpts_lower = baseline_data['y_cpts_lower']
        print(f"Successfully loaded baseline control points from '{baseline_cpts_file}'")
    except FileNotFoundError:
        print(f"Error: Could not find '{baseline_cpts_file}'. Please run the parameterization script first.")
        return

    # 2. Generate the full airfoil coordinates
    x_coords, y_coords = generate_coords_from_cpts(
        x_cpts, y_cpts_upper, y_cpts_lower
    )

    # --- ADDITION: Calculate the area of the baseline airfoil ---
    baseline_area = calculate_area(x_coords, y_coords)
    print(f"Calculated Baseline Area: {baseline_area:.8f}")
    # --- END ADDITION ---

    # 3. Run the XFOIL simulation
    print("Running XFOIL analysis on the baseline geometry...")
    polar = run_xfoil(
        x_coords, y_coords, 
        airfoil_name="baseline_8M", 
        reynolds=500000, 
        alpha_start=0, 
        alpha_end=20, 
        alpha_step=1.0
    )

    # 4. Calculate the reward for the baseline airfoil
    if polar is not None:
        # --- MODIFICATION: Pass the required area arguments to the reward function ---
        # For the baseline, the "new_area" and "baseline_area" are the same.
        baseline_reward = calculate_reward(polar, new_area=baseline_area, baseline_area=baseline_area)
        # --- END MODIFICATION ---

        print("\n--- Baseline Performance Results ---")
        print(f"Baseline Reward Score: {baseline_reward:.4f}")
        print("\nBaseline Polar Data:")
        print(polar.to_string())
    else:
        print("\n--- XFOIL analysis failed for the baseline airfoil. ---")


if __name__ == '__main__':
    evaluate_baseline_performance()