import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def generate_control_point_distribution(num_points=12, x_min=0.0, x_max=1.0):
    """
    Generates the x-coordinates for control points using cosine spacing,
    scaled and shifted to fit the specified x_min and x_max range.

    Args:
        num_points (int): The total number of control points.
        x_min (float): The minimum x-coordinate (the leading edge).
        x_max (float): The maximum x-coordinate (the trailing edge).

    Returns:
        np.ndarray: An array of scaled x-coordinates for the control points.
    """
    # --- MODIFICATION START ---
    # First, generate the normalized distribution from 0 to 1
    n = np.arange(num_points)
    normalized_dist = 0.5 * (1 - np.cos(n / (num_points - 1) * np.pi))
    
    # Now, scale and shift this distribution to the actual airfoil's range
    x_cpts = x_min + (x_max - x_min) * normalized_dist
    # --- MODIFICATION END ---
    return x_cpts

def fit_bspline_to_file(filepath, num_control_points=12):
    """
    Loads airfoil data, fits B-splines, and returns the results.
    This version correctly handles airfoils not starting at x=0.

    Args:
        filepath (str): Path to the airfoil .dat file.
        num_control_points (int): The number of control points to use for the fit.

    Returns:
        tuple or None: A tuple containing all data needed for saving and plotting.
    """
    try:
        original_coords = np.loadtxt(filepath, skiprows=1)
        print(f"Successfully loaded airfoil data from '{filepath}'")
    except Exception as e:
        print(f"Error: Could not load file '{filepath}'.")
        return None

    le_index = np.argmin(original_coords[:, 0])
    x_upper_raw = np.flip(original_coords[:le_index + 1, 0])
    y_upper_raw = np.flip(original_coords[:le_index + 1, 1])
    x_lower_raw = original_coords[le_index:, 0]
    y_lower_raw = original_coords[le_index:, 1]

    # --- MODIFICATION START ---
    # 1. Find the actual min and max x-coordinates from the data
    x_le = original_coords[le_index, 0]
    y_le = original_coords[le_index, 1] # y-value at the leading edge
    x_te = np.max(original_coords[:, 0])
    print(f"Detected airfoil x-range: [{x_le:.6f}, {x_te:.6f}]")

    # 2. Generate the control point x-locations for this specific range
    x_cpts = generate_control_point_distribution(num_control_points, x_min=x_le, x_max=x_te)
    # --- MODIFICATION END ---

    tck_upper = interpolate.splrep(x_upper_raw, y_upper_raw, k=3, s=0.0)
    tck_lower = interpolate.splrep(x_lower_raw, y_lower_raw, k=3, s=0.0)
    print("Found best-fit mathematical spline for upper and lower surfaces.")

    spline_upper_func = interpolate.BSpline(tck_upper[0], tck_upper[1], tck_upper[2])
    spline_lower_func = interpolate.BSpline(tck_lower[0], tck_lower[1], tck_lower[2])
    y_cpts_upper = spline_upper_func(x_cpts)
    y_cpts_lower = spline_lower_func(x_cpts)
    
    # --- MODIFICATION START ---
    # 3. Pin the boundary conditions using the actual LE and TE data
    # Pin the leading edge control point to its actual (x, y) location
    y_cpts_upper[0] = y_le
    y_cpts_lower[0] = y_le
    
    # Use the exact trailing edge formula from the paper
    y_upper_TE = y_upper_raw[-1]
    y_lower_TE = y_lower_raw[-1]
    TE_base = y_upper_TE - y_lower_TE
    half_TE_base = TE_base / 2.0
    y_cpts_upper[-1] = half_TE_base
    y_cpts_lower[-1] = -half_TE_base
    
    print("Final control point y-coordinates calculated and boundaries pinned.")
    print(f"Applied paper's TE formula: TE_base={TE_base:.6f}, Final CPs at y=±{half_TE_base:.6f}")
    # --- MODIFICATION END ---

    return original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower

def save_control_points(filename, x_cpts, y_cpts_upper, y_cpts_lower):
    """Saves the calculated control points to a compressed NumPy file."""
    np.savez(
        filename,
        x_cpts=x_cpts,
        y_cpts_upper=y_cpts_upper,
        y_cpts_lower=y_cpts_lower
    )
    print(f"Baseline control points saved to '{filename}'")

def plot_parameterization(original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower):
    """
    Creates a plot showing the 8M airfoil, the fitted B-spline, and the control points.
    """
    # Generate a high-resolution version of the B-spline curve for smooth plotting
    x_fine = np.linspace(np.min(original_coords[:, 0]), np.max(original_coords[:, 0]), 400)
    y_upper_fine = interpolate.splev(x_fine, tck_upper)
    y_lower_fine = interpolate.splev(x_fine, tck_lower)

    plt.figure(figsize=(16, 8))
    
    # Plot the original 8M data points as a reference
    plt.plot(original_coords[:, 0], original_coords[:, 1], 'o', color='gray', markersize=3, label='8M Airfoil Data Points')
    # Plot the smooth B-spline curve that represents the data
    plt.plot(x_fine, y_upper_fine, 'b-', linewidth=2, label='Fitted B-Spline Curve')
    plt.plot(x_fine, y_lower_fine, 'b-', linewidth=2)
    # Plot the control points that define the B-spline
    plt.plot(x_cpts, y_cpts_upper, 'ro-', label='Upper Surface Control Points')
    plt.plot(x_cpts, y_cpts_lower, 'go-', label='Lower Surface Control Points')
    
    plt.title('8M Airfoil B-Spline Parameterization', fontsize=16)
    plt.xlabel('x/c', fontsize=12)
    plt.ylabel('y/c', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # --- CONFIGURATION ---
    AIRFOIL_DAT_FILE = 'peak_8M_airfoil.dat'
    CONTROL_POINTS_OUTPUT_FILE = 'baseline_control_points.npz'
    NUM_CONTROL_POINTS = 8

    # --- MAIN EXECUTION ---
    # 1. Fit the B-spline to the data in the .dat file
    fit_results = fit_bspline_to_file(AIRFOIL_DAT_FILE, NUM_CONTROL_POINTS)

    # 2. If fitting was successful, save the points and plot the result
    if fit_results:
        # Unpack the results tuple
        original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower = fit_results
        
        # Save the calculated control points for the DRL environment
        save_control_points(CONTROL_POINTS_OUTPUT_FILE, x_cpts, y_cpts_upper, y_cpts_lower)
        
        # Create the verification plot
        plot_parameterization(original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower)