import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def generate_x_cpts_from_parametric_map(num_points=8, clustering_factor_k=2.0):
    """
    Generates the x-coordinates for control points based on the robust
    parametric mapping x = u^k, as described by Rajnarayan et al.

    This method creates a stable link between the B-spline's parametric space
    and the physical space, naturally clustering points at the leading edge.

    Args:
        num_points (int): The total number of control points for one surface.
        clustering_factor_k (float): The exponent for the mapping x = u^k.
                                     k=1 is uniform, k=2 is standard, k>2 is more LE-focused.

    Returns:
        np.ndarray: A normalized array of derived x-coordinates (from 0 to 1).
    """
    # 1. Define the B-spline's internal parametric space 'u' (uniform from 0 to 1)
    #    and the desired physical space 'x' using the mapping.
    u_map = np.linspace(0, 1, 101)
    x_map = u_map ** clustering_factor_k

    # 2. We need to find the control points of a B-spline that represents this mapping.
    #    A cubic B-spline (k=3) is standard and robust.
    #    The knots must be uniform in the parametric (u) space.
    num_internal_knots = num_points - 3
    knots = np.pad(np.linspace(0, 1, num_internal_knots + 2), (3,3), 'constant', constant_values=(0,1))

    # 3. Use scipy's spline fitting to create the B-spline object for the map.
    tck = interpolate.splrep(u_map, x_map, t=knots[4:-4], k=3, s=0)
    
    # 4. The x-locations of the control points of this mapping spline are our final x_cpts.
    #    The control points are the second element of the tck tuple.
    x_cpts = np.sort(tck[1][:num_points])
    
    # Ensure start and end are exactly 0 and 1
    x_cpts[0] = 0.0
    x_cpts[-1] = 1.0
    
    return x_cpts

def fit_bspline_to_file(filepath, num_control_points=8):
    """
    Loads airfoil data, fits B-splines using the robust Rajnarayan
    parametrization, and returns the results.

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

    # 1. Find the actual min and max x-coordinates from the data
    x_le = original_coords[le_index, 0]
    y_le = original_coords[le_index, 1] # y-value at the leading edge
    x_te = np.max(original_coords[:, 0])
    print(f"Detected airfoil x-range: [{x_le:.6f}, {x_te:.6f}]")

    # --- CORRECTED LOGIC HERE ---
    # 2. Generate the NORMALIZED (0 to 1) control point distribution using the new function
    normalized_x_cpts = generate_x_cpts_from_parametric_map(num_control_points, clustering_factor_k=2.0)
    
    # 3. Scale this robust distribution to the actual airfoil's physical range
    x_cpts = x_le + (x_te - x_le) * normalized_x_cpts
    # --- END CORRECTION ---
    
    print("Generated x-control points using Rajnarayan's parametric mapping.")

    # Fit B-splines to the raw y-data at the newly defined x_cpts locations
    tck_upper = interpolate.splrep(x_upper_raw, y_upper_raw, k=3, s=0.0)
    tck_lower = interpolate.splrep(x_lower_raw, y_lower_raw, k=3, s=0.0)
    print("Found best-fit mathematical spline for upper and lower surfaces.")

    # Evaluate the splines at the x_cpts to get the corresponding y_cpts
    y_cpts_upper = interpolate.splev(x_cpts, tck_upper)
    y_cpts_lower = interpolate.splev(x_cpts, tck_lower)
    
    # Pin the boundary conditions using the actual LE and TE data
    y_cpts_upper[0] = y_le
    y_cpts_lower[0] = y_le
    
    # For a symmetrical airfoil with a sharp TE at y=0, this is sufficient.
    # For more complex cases, one might interpolate the TE y-values.
    y_upper_TE_interp = interpolate.splev(x_te, tck_upper)
    y_lower_TE_interp = interpolate.splev(x_te, tck_lower)
    y_cpts_upper[-1] = y_upper_TE_interp
    y_cpts_lower[-1] = y_lower_TE_interp
    
    print("Final control point y-coordinates calculated and boundaries pinned.")

    return original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower

def save_control_points(filename, x_cpts, y_cpts_upper, y_cpts_lower):
    """Saves the calculated control points to a compressed NumPy file."""
    np.savez(
        filename,
        x_cpts=x_cpts,
        y_cpts_upper=y_cpts_upper,
        y_cpts_lower=y_cpts_lower
    )
    print(f"\nBaseline control points saved to '{filename}'")

def plot_parameterization(original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower):
    """
    Creates a plot showing the 8M airfoil, the fitted B-spline, and the control points.
    """
    x_fine = np.linspace(np.min(original_coords[:, 0]), np.max(original_coords[:, 0]), 400)
    y_upper_fine = interpolate.splev(x_fine, tck_upper)
    y_lower_fine = interpolate.splev(x_fine, tck_lower)

    plt.figure(figsize=(16, 8))
    plt.plot(original_coords[:, 0], original_coords[:, 1], 'o', color='gray', markersize=3, label='Original Airfoil Data Points')
    plt.plot(x_fine, y_upper_fine, 'b-', linewidth=2, label='Fitted B-Spline Curve')
    plt.plot(x_fine, y_lower_fine, 'b-', linewidth=2)
    plt.plot(x_cpts, y_cpts_upper, 'ro-', label='Upper Surface Control Points')
    plt.plot(x_cpts, y_cpts_lower, 'go-', label='Lower Surface Control Points')
    
    plt.title('Airfoil B-Spline Parameterization (Rajnarayan Method)', fontsize=16)
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
    NUM_CONTROL_POINTS = 8 # Keep at 8 per surface, as determined before

    # --- MAIN EXECUTION ---
    print("--- Starting Airfoil Parameterization using Rajnarayan's Method ---")
    fit_results = fit_bspline_to_file(AIRFOIL_DAT_FILE, NUM_CONTROL_POINTS)

    if fit_results:
        original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower = fit_results
        
        save_control_points(CONTROL_POINTS_OUTPUT_FILE, x_cpts, y_cpts_upper, y_cpts_lower)
        
        print("\nVerification plot will now be displayed...")
        plot_parameterization(original_coords, x_cpts, y_cpts_upper, y_cpts_lower, tck_upper, tck_lower)