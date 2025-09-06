# File: calculate_baseline_area.py
import numpy as np
from airfoil_geometry import generate_coords_from_cpts

def calculate_area(x_coords, y_coords):
    """ Calculates the area of a polygon using the Shoelace formula. """
    # Ensure the polygon is closed
    x = np.append(x_coords, x_coords[0])
    y = np.append(y_coords, y_coords[0])
    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return area

if __name__ == '__main__':
    baseline_cpts_file = 'baseline_control_points.npz'
    print(f"--- Calculating Area for Baseline Airfoil from '{baseline_cpts_file}' ---")

    baseline_data = np.load(baseline_cpts_file)
    x_cpts = baseline_data['x_cpts']
    y_cpts_upper = baseline_data['y_cpts_upper']
    y_cpts_lower = baseline_data['y_cpts_lower']

    x_coords, y_coords = generate_coords_from_cpts(x_cpts, y_cpts_upper, y_cpts_lower)
    
    baseline_area = calculate_area(x_coords, y_coords)
    
    print("\n----------------------------------------------------")
    print(f"  Baseline Airfoil Area: {baseline_area:.8f}")
    print("----------------------------------------------------")
    print("\nCopy this value into your drl_environment.py file.")