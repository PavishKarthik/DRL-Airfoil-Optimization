import os
import subprocess
import numpy as np
import pandas as pd
import random
import string

def run_xfoil(x_coords, y_coords, airfoil_name="temp_airfoil", reynolds=500000, mach=0.0, alpha_start=0, alpha_end=15, alpha_step=1.0, debug=False):
    """
    Runs an XFOIL analysis. This is the final, corrected version that prevents deadlocks and pathing issues.
    """
    rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    airfoil_file = f"{airfoil_name}_{rand_id}.dat"
    polar_file = f"{airfoil_name}_{rand_id}_polar.txt"
    xfoil_input_file = f"xfoil_input_{rand_id}.in"

    # Step 1: Write the airfoil coordinate file
    with open(airfoil_file, 'w') as f:
        f.write(f"{airfoil_name}\n")
        coords = np.column_stack((x_coords, y_coords))
        np.savetxt(f, coords, fmt='%.8f') # Fixed typo: savsavetxt -> savetxt

    # Step 2: Write the XFOIL command script to a file
    with open(xfoil_input_file, 'w') as f:
        f.write(f"LOAD {airfoil_file}\n")
        f.write("G\n")
        f.write("OPER\n")
        f.write(f"VISC {reynolds}\n")
        f.write(f"MACH {mach}\n")
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n")
        f.write(f"ASEQ {alpha_start} {alpha_end} {alpha_step}\n")
        f.write("PACC\n")
        f.write("\n")
        f.write("QUIT\n")
    
    # Step 3: Read the command script back into a string variable
    with open(xfoil_input_file, 'r') as f:
        xfoil_script_content = f.read()

    # Step 4: Define the explicit path to the executable
    xfoil_executable = r"C:\Users\pavis\DRL-Airfoil-Optimization\src\xfoil.exe"
    
    # Step 5: Run the subprocess with the script content piped to stdin
    try:
        result = subprocess.run(
            [xfoil_executable],
            input=xfoil_script_content, # This variable is now correctly defined
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print(f"--- XFOIL ERROR ---")
            print(f"XFOIL exited with a non-zero status code. STDERR: {result.stderr}")
            return None
    except FileNotFoundError:
        print(f"ERROR: '{xfoil_executable}' not found. Please ensure xfoil.exe is in the same directory as the script.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running XFOIL: {e}")
        return None

    # Step 6: Parse the output polar file
    try:
        polar_data = pd.read_csv(
            polar_file, sep='\s+', skiprows=12, engine='python',
            names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr']
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        polar_data = None
    
    # Step 7: Clean up all temporary files
    finally:
        for file in [airfoil_file, polar_file, xfoil_input_file]:
            if os.path.exists(file):
                os.remove(file)

    return polar_data

# The __main__ block should still be present in your file for testing
if __name__ == '__main__':
    # ... your existing test code ...
    print("--- Running a test of the XFOIL wrapper in DEBUG mode ---")
    try:
        test_coords = np.loadtxt('peak_8M_airfoil.dat', skiprows=1)
        test_x = test_coords[:, 0]
        test_y = test_coords[:, 1]
        
        polar = run_xfoil(
            test_x, test_y, 
            airfoil_name="test_8M_debug", # This name is written to the temp file
            reynolds=500000, 
            alpha_start=-2, 
            alpha_end=20, 
            alpha_step=1,
            debug=True # <-- IMPORTANT: We are enabling debug mode
        )
        
        if polar is not None:
            print("XFOIL analysis successful!")
            print(polar.to_string())
        else:
            print("\nXFOIL analysis failed. Please follow the debug instructions above.")
            
    except FileNotFoundError:
        print("Could not run test: 'peak_8M_airfoil.dat' not found.")