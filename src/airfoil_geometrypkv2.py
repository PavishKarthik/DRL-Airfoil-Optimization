import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_naca_63_4_021():
    """
    Generates the coordinates for the NACA 634-021 airfoil based directly
    on the tabulated data from "Theory of Wing Sections".
    Returns the raw upper surface coordinates.
    """
    # Raw data transcribed directly from the table for "NACA 63A-021 Basic Thickness Form".
    x_percent = np.array([0, 0.5, 0.75, 1.25, 2.5, 5.0, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    yt_percent = np.array([0, 1.583, 1.937, 2.527, 3.577, 5.065, 6.182, 7.080, 8.441, 9.410, 10.053, 10.412, 10.500, 10.298, 9.854, 9.206, 8.390, 7.441, 6.396, 5.290, 4.160, 3.054, 2.021, 1.113, 0.392, 0])

    x_base = x_percent / 100.0
    yt_base = yt_percent / 100.0

    # Scale to 21% thickness
    max_thickness_in_table = 0.105
    desired_max_thickness = 0.21
    yt_scaled = yt_base * (desired_max_thickness / max_thickness_in_table)

    return x_base, yt_scaled

def create_tubercled_peak_section(x_base, yt_base, amplitude):
    """
    Creates the 2D peak cross-section using a robust implementation of the
    non-linear shearing transformation from Lohry et al. (2012).
    This version correctly integrates the transformation to avoid numerical errors.
    """
    # --- Setup based on Lohry et al. (2012) ---
    C0 = 1.0  # Unmodified chord length
    C_prime = 1.0 + amplitude # New, scaled chord length of the peak section

    # The transformation is only applied to the forward 30% of the chord (eta <= 0.3)
    # We need to find which of our x_base points fall into this category.
    # Note: eta is the normalized coordinate on the *original* chord.
    eta = x_base / C0
    mask = eta <= 0.3
    
    eta_front = eta[mask]
    eta_rear = eta[~mask]

    # --- Implement the Non-Linear Transformation from Eq. 6 ---
    # d(eta')/d(eta) = 1 + B*sin(pi*eta)
    # The constant B is chosen to ensure the total length of the front section
    # is stretched correctly. The integration of the stretching part must equal
    # the total stretch amount, which is (C_prime - C0) * 0.3.
    # After solving the integral of B*sin(pi*eta) from 0 to 0.3, we can find B.
    
    # Total stretch required for the front 30% of the airfoil
    total_stretch = C_prime - C0
    
    # The integral of sin(pi*eta) from 0 to 0.3 gives a constant factor
    integral_factor = (-np.cos(0.3 * np.pi) / np.pi) + (np.cos(0) / np.pi)
    
    # We can now solve for B
    B = total_stretch / integral_factor if integral_factor != 0 else 0

    # Eq. 7 is the integral of Eq. 6: eta' = eta + integral(B*sin(pi*eta))
    # The integral of B*sin(pi*eta) is -B/pi * cos(pi*eta)
    eta_prime_front = eta_front - (B / np.pi) * (np.cos(np.pi * eta_front) - 1)

    # The new x-coordinates for the front section are eta_prime scaled by the original chord
    x_new_front = eta_prime_front * C0

    # The rear section is linearly shifted to connect smoothly with the front
    # Find the position of the last point of the transformed front section
    x_junction_point = x_new_front[-1]
    
    # The rear section starts at x=0.3 on the original airfoil
    x_rear_original = x_base[~mask]
    
    # We shift the entire rear section to start at our new junction point
    x_new_rear = x_rear_original - 0.3 + x_junction_point

    # Combine the transformed front and the shifted rear
    x_new_upper = np.concatenate((x_new_front, x_new_rear))
    y_new_upper = yt_base

    # --- Assemble the full airfoil ---
    x_upper_flipped = np.flip(x_new_upper)
    y_upper_flipped = np.flip(y_new_upper)

    x_lower = x_new_upper[1:]
    y_lower = -y_new_upper[1:]

    x_final = np.concatenate((x_upper_flipped, x_lower))
    y_final = np.concatenate((y_upper_flipped, y_lower))

    return x_final, y_final

def verify_t_over_c(x, y):
    """
    Calculates the maximum thickness to chord ratio (t/c) of an airfoil.
    """
    # Find the leading edge point to correctly split upper and lower surfaces
    le_index = np.argmin(x)
    x_upper = x[:le_index + 1]
    y_upper = y[:le_index + 1]
    x_lower = x[le_index:]
    y_lower = y[le_index:]

    # Interpolate the lower surface y-values at the upper surface x-locations
    y_lower_interp = np.interp(np.flip(x_upper), np.flip(x_lower), np.flip(y_lower))
    
    # Thickness is the difference between the y-values
    thickness = np.flip(y_upper) - y_lower_interp
    max_thickness = np.max(thickness)
    
    # Chord is the total length in the x-direction
    chord = np.max(x) - np.min(x)
    
    t_over_c = max_thickness / chord
    return t_over_c

def verify_le_radius(x, y):
    """
    Estimates the leading edge radius by fitting a circle to the 3 nose points.
    """
    # Find the leading edge point (the point with the minimum x value)
    le_index = np.argmin(x)
    
    p1 = (x[le_index], y[le_index])             # LE point
    p2 = (x[le_index - 1], y[le_index - 1])     # Point just before LE on upper surface
    p3 = (x[le_index + 1], y[le_index + 1])     # Point just after LE on lower surface

    # Formula for the radius of a circle defined by three points
    A = p1[0] * (p2[1] - p3[1]) - p1[1] * (p2[0] - p3[0]) + p2[0] * p3[1] - p3[0] * p2[1]
    B = (p1[0]**2 + p1[1]**2) * (p3[1] - p2[1]) + (p2[0]**2 + p2[1]**2) * (p1[1] - p3[1]) + (p3[0]**2 + p3[1]**2) * (p2[1] - p1[1])
    C = (p1[0]**2 + p1[1]**2) * (p2[0] - p3[0]) + (p2[0]**2 + p2[1]**2) * (p3[0] - p1[0]) + (p3[0]**2 + p3[1]**2) * (p1[0] - p2[0])
    
    if abs(A) < 1e-9: return float('inf')
        
    radius = np.sqrt(B**2 + C**2) / (2 * abs(A))
    return radius

def plot_airfoil(x, y, title="Airfoil Shape"):
    """
    Plots the airfoil coordinates for visual inspection.
    """
    plt.figure(figsize=(12, 3))
    plt.plot(x, y, 'o-', label='Airfoil Points', markersize=3)
    plt.title(title)
    plt.xlabel("x / c (Chord)")
    plt.ylabel("y / c (Chord)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def save_airfoil_to_dat(x, y, filename, project_root):
    """
    Saves the airfoil coordinates to a .dat file in the format required by XFOIL.
    """
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    coords = np.vstack((x, y)).T
    np.savetxt(filepath, coords, header=filename, comments='', fmt='%.6f')
    print(f"Saved to: {filepath}")

# This block runs ONLY when the script is executed directly
if __name__ == '__main__':
    # Define the project root directory to save files correctly
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # --- Part 1: Generate and analyze the baseline airfoil ---
    x_baseline_raw, y_baseline_raw = generate_naca_63_4_021()
    
    # Assemble the full, symmetric baseline airfoil from the raw upper surface data
    x_baseline_full = np.concatenate((np.flip(x_baseline_raw), x_baseline_raw[1:]))
    y_baseline_full = np.concatenate((np.flip(y_baseline_raw), -y_baseline_raw[1:]))
    
    print("--- Baseline NACA 634-021 ---")
    print(f"Max t/c: {verify_t_over_c(x_baseline_full, y_baseline_full):.4f}")
    print(f"LE Radius: {verify_le_radius(x_baseline_full, y_baseline_full):.6f}")
    
    plot_airfoil(x_baseline_full, y_baseline_full, title="Baseline NACA 634-021")
    save_airfoil_to_dat(x_baseline_full, y_baseline_full, "naca634021_baseline.dat", PROJECT_ROOT)
    print("-" * 30)

    # --- Part 2: Generate and analyze the "8M" tubercled airfoil ---
    amplitude_8M = 0.05 # A/c = 0.05 for the "M" configuration from Johari et al.
    x_8M, y_8M = create_tubercled_peak_section(x_baseline_raw, y_baseline_raw, amplitude_8M)
    
    print("--- 8M Tubercled Airfoil ---")
    print(f"Max t/c: {verify_t_over_c(x_8M, y_8M):.4f}")
    print(f"LE Radius: {verify_le_radius(x_8M, y_8M):.6f}")

    plot_airfoil(x_8M, y_8M, title="8M Tubercled Airfoil (Non-Linear Shear)")
    save_airfoil_to_dat(x_8M, y_8M, "naca634021_8M_nonlinear.dat", PROJECT_ROOT)