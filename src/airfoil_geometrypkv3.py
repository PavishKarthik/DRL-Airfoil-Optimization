import numpy as np
import matplotlib.pyplot as plt

def genrate_n634021():
    """
    Genrates the NACA 634-021 airfoil by using the "NACA 634 021 Basis Thickness form" table from 
    "Theory of Wing sections" by Ira H.Abbott and Albert E.Von Doenhoff.
    This function is UNCHANGED as requested.
    """
    x_percent_of_c = np.array([0, 0.5, 0.75, 1.25, 2.5, 5.0, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    yt_percent_of_c = np.array([0, 1.583, 1.937, 2.527, 3.577, 5.065, 6.182, 7.080, 8.441, 9.410, 10.053, 10.412, 10.500, 10.298, 9.854, 9.206,
                            8.390, 7.441, 6.396, 5.290, 4.160, 3.054, 2.021, 1.113, 0.392, 0])
    
    xt = x_percent_of_c/100
    yt = yt_percent_of_c/100

    #scaling 21% thickness
    max_thickness = 0.21
    yt = (yt/0.105)*max_thickness
    #Conventionally XFOIL like evaluators require airfoil coordinates starting from TE and move in clockwise direction and end in TE
    x_upper = np.flip(xt)
    y_upper = np.flip(yt)

    #Excluding LE (0,0)
    x_lower = xt[1:]
    y_lower = -yt[1:]

    x = np.concatenate((x_upper,x_lower))
    y = np.concatenate((y_upper,y_lower))

    return x,y

# --- NEW HELPER FUNCTIONS FOR ANALYSIS ---
def get_max_thickness(x_coords, y_coords):
    """Calculates the maximum thickness and its location."""
    # Find the upper and lower surfaces based on the ordering from genrate_n634021
    # The first half is the upper surface (TE to LE)
    num_points_upper = len(x_coords) // 2 + 1
    upper_y = y_coords[:num_points_upper]
    upper_x = x_coords[:num_points_upper]
    
    # Simple approximation: max thickness is max y on upper surface * 2
    max_thick = 2 * np.max(upper_y)
    location_index = np.argmax(upper_y)
    x_location = upper_x[location_index]
    
    return max_thick, x_location

def get_le_radius(x_coords, y_coords):
    """Estimates the leading-edge radius using 3 points near the LE."""
    # Find the leading edge point (minimum x value)
    le_index = np.argmin(x_coords)
    
    # Get the LE point and its neighbors on the upper and lower surfaces
    p1 = np.array([x_coords[le_index + 1], y_coords[le_index + 1]]) # Point on lower surface
    p2 = np.array([x_coords[le_index], y_coords[le_index]])     # LE point
    p3 = np.array([x_coords[le_index - 1], y_coords[le_index - 1]]) # Point on upper surface
    
    # Use the formula for the radius of a circle through three points
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p3 - p1)
    
    # Using Heron's formula for area to avoid issues with vertical lines
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    if area == 0:
        return float('inf')
        
    radius = (a * b * c) / (4 * area)
    return radius


if __name__ == '__main__':
    # 1. Get the baseline coordinates from your function
    base_x, base_y = genrate_n634021()

    # --- SETUP FOR PEAK AIRFOIL TRANSFORMATION ---
    print("--- Calculating Peak Airfoil ---")
    airfoil_type = 'peak'
    A = 0.05
    if airfoil_type == 'peak':
        x_LE = -A
    # Add other types here if needed
    # elif airfoil_type == 'trough':
    #     x_LE = A
        
    C_prime = 1.0 - x_LE
    eta_transition = 0.35
    D_denominator = np.cos(eta_transition * np.pi) - C_prime * np.cos(np.pi * (-x_LE) / C_prime)

    # 2. Loop through each baseline x-coordinate to create the new airfoil
    x_8M_peak = []
    for x0_point in base_x:
        # This is the core logic from your function, now applied point-by-point
        # and corrected based on the final derived formula.
        if x0_point <= eta_transition:
            # Apply transformation to the front section
            x_new = x0_point + (x_LE / D_denominator) * \
                    (np.cos(eta_transition * np.pi) - C_prime * np.cos(np.pi * (x0_point - x_LE) / C_prime))
        else:
            # The back section is not changed
            x_new = x0_point
        x_8M_peak.append(x_new)

    # 3. Combine the new x-coordinates with the original y-coordinates
    x_8M_peak = np.array(x_8M_peak)
    y_8M_peak = base_y # Y coordinates are unchanged

    # --- Geometric Analysis ---
    print("\n--- Geometric Analysis ---")
    base_thick, base_thick_loc = get_max_thickness(base_x, base_y)
    base_le_rad = get_le_radius(base_x, base_y)
    print(f"Baseline Airfoil: Max Thickness={base_thick:.4f}c at x={base_thick_loc:.3f}c, LE Radius={base_le_rad:.6f}c")

    peak_thick, peak_thick_loc = get_max_thickness(x_8M_peak, y_8M_peak)
    peak_le_rad = get_le_radius(x_8M_peak, y_8M_peak)
    print(f"Peak Airfoil:     Max Thickness={peak_thick:.4f}c at x={peak_thick_loc:.3f}c, LE Radius={peak_le_rad:.6f}c")

    # 4. Plot the results
    plt.figure(figsize=(15, 10))
    plt.plot(base_x, base_y, 'k--', label=f'Baseline NACA 634-021', linewidth=2)
    plt.plot(x_8M_peak, y_8M_peak, 'b-', label='Transformed Peak Airfoil (8M)')

    # Add a marker for the transition point to verify
    plt.axvline(x=0.35, color='gray', linestyle=':', label='Transition Point (35% Chord)')

    plt.title('Airfoil Transformation using Lohry et al. Method')
    plt.xlabel('Chordwise Position (x/c)')
    plt.ylabel('Thickness Position (y/c)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()