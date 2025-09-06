import numpy as np
import matplotlib.pyplot as plt

def genrate_n634021():
    """
    Genrates the NACA 634-021 airfoil by using the "NACA 634 021 Basis Thickness form" table from 
    "Theory of Wing sections" by Ira H.Abbott and Albert E.Von Doenhoff

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

def create_8M_airfoil(base_x):

    z=1/16 # so that sin becomes 1 for peak 
    A = 0.05
    lmda = 0.25
    k = (2*np.pi)/lmda


    x_LE = -A*np.sin(k*z) # neg sign is used as we are moving the airfoil towards left 
    xo_LE = 0
    Co = 1
    Eta_transition = 0.35

    C_prime = 1-x_LE
    Eta = (base_x - x_LE)/C_prime
    # NOTE: The logic here contains known issues but is kept as is per instruction.
    ko = ( xo_LE - x_LE)/(((((C_prime/Co)-1))/np.pi)*((C_prime*np.cos(np.pi*(xo_LE-x_LE)/C_prime))-np.cos(0.3*np.pi)))
    k1 = x_LE-xo_LE +(ko/np.pi)*((C_prime/Co)-1)*C_prime*np.cos(np.pi*(base_x-x_LE)/C_prime)
    B = ko*((C_prime/Co) - 1)

    Eta_prime = np.where(base_x < Eta_transition, Eta - (B/np.pi)*np.cos(np.pi*Eta), Eta)
    
    # NOTE: This logic contains known issues but is kept as is per instruction.
    x_new_if = base_x - C_prime*(ko/np.pi)*((C_prime/Co)-1)*np.cos(np.pi*(base_x-x_LE)/C_prime) + k1
    x_new_else = base_x

    x_new = np.where(Eta_prime < Eta_transition,x_new_if,x_new_else)
    

    return x_new

# --- HELPER FUNCTIONS FOR GEOMETRIC ANALYSIS ---

def get_max_thickness(x_coords, y_coords):
    """Calculates the maximum thickness and its location."""
    num_points_upper = len(x_coords) // 2 + 1
    upper_y = y_coords[:num_points_upper]
    upper_x = x_coords[:num_points_upper]
    
    max_thick = 2 * np.max(upper_y)
    location_index = np.argmax(upper_y)
    x_location = upper_x[location_index]
    
    return max_thick, x_location

def get_le_radius(x_coords, y_coords):
    """Estimates the leading-edge radius using 3 points near the LE."""
    le_index = np.argmin(x_coords)
    
    p1 = np.array([x_coords[le_index + 1], y_coords[le_index + 1]])
    p2 = np.array([x_coords[le_index], y_coords[le_index]])
    p3 = np.array([x_coords[le_index - 1], y_coords[le_index - 1]])
    
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p3 - p1)
    
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c)) + 1e-12
        
    radius = (a * b * c) / (4 * area)
    return radius

# --- NEW HELPER FUNCTION TO SAVE COORDINATES ---
def save_airfoil_to_dat(filename, x_coords, y_coords, airfoil_name=""):
    """
    Saves airfoil coordinates to a .dat file in the standard XFOIL/Selig format.
    
    Args:
        filename (str): The name of the file to save (e.g., 'peak_8M.dat').
        x_coords (np.ndarray): The x-coordinates of the airfoil.
        y_coords (np.ndarray): The y-coordinates of the airfoil.
        airfoil_name (str): The name to write on the first line of the file.
    """
    # Combine x and y into a single (N, 2) array
    coords = np.column_stack((x_coords, y_coords))
    
    # The standard format is TE -> Upper -> LE -> Lower -> TE.
    # Your genrate_n634021() function already produces this order.
    
    try:
        with open(filename, 'w') as f:
            if airfoil_name:
                f.write(f"{airfoil_name}\n")
            
            # Use numpy's savetxt for easy and clean formatting
            np.savetxt(f, coords, fmt='%.8f')
        print(f"Successfully saved airfoil to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")

def plot(base_x,base_y,x_8M,y_8M):
    # 7. Plot the original airfoil and the newly created one
    plt.figure(figsize=(15, 10))
    plt.plot(base_x, base_y, 'k--', label='Baseline NACA 634-021', linewidth=2)
    plt.plot(x_8M, y_8M, 'b-', label='Transformed 8M Airfoil (from your function)')
    
    plt.title('Airfoil Plot')
    plt.xlabel('Chordwise Position (x/c)')
    plt.ylabel('Thickness Position (y/c)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 1. Get the baseline coordinates from your function
    base_x, base_y = genrate_n634021()

    # 2. Create an empty list to store the new x-coordinates
    
    x_8M = create_8M_airfoil(base_x=base_x)
    y_8M = base_y
    print("Transformation complete.")

    # Perform geometric analysis
    print("\n--- Geometric Analysis ---")
    base_thick, base_thick_loc = get_max_thickness(base_x, base_y)
    base_le_rad = get_le_radius(base_x, base_y)
    print(f"Baseline Airfoil: Max Thickness={base_thick:.4f}c at x={base_thick_loc:.3f}c, LE Radius={base_le_rad:.6f}c")

    peak_thick, peak_thick_loc = get_max_thickness(x_8M, y_8M)
    peak_le_rad = get_le_radius(x_8M, y_8M)
    print(f"Peak Airfoil:     Max Thickness={peak_thick:.4f}c at x={peak_thick_loc:.3f}c, LE Radius={peak_le_rad:.6f}c")
    
    # Save the new airfoil coordinates to a .dat file
    print("\n--- Saving Airfoil Data ---")
    save_airfoil_to_dat('peak_8M_airfoil.dat', x_8M, y_8M, airfoil_name="Peak 8M Transformed Airfoil")
    
    plot(base_x,base_y,x_8M,y_8M)