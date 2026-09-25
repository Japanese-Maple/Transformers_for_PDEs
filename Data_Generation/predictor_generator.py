import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
parent_root = os.path.dirname(project_root)
for path in (project_root, parent_root):
    if path not in sys.path:
        sys.path.append(path)

import numpy as np
from matplotlib.path import Path

from data_utils import (
    load_mesh_data,
    sample_mu_values,
    sample_mu_parameter_locations,
    basis_generator,
    save_parameters_data
)

#____________________________________________________________________________________________________________________________
# MAIN CONTROLS

num_samples      = 700
k_params         = 9
minimum_distance = 0.5
mu_range         = [0.1, 100.0]

#____________________________________________________________________________________________________________________________
# LOADING MESH

mesh_data_path = os.path.join(project_root, 'Data_Generation/Data', 'ex_dev.npz')
p_fine, e_fine, t_fine, p_coarse, e_coarse, t_coarse, b_nodes = load_mesh_data(mesh_data_path)

main_region = np.array([
    [ 0.95,  1.0],
    [ 0.95, -1/3],
    [ 1.05, -1/3],
    [ 1.05,  1.0],
    [ 2.00,  1.0],
    [ 2.00,  1/3],
    [ 2.50,  1/3],
    [ 2.50, -1/3],
    [ 2.00, -1/3],
    [ 2.00, -1.0],
    [ 0.05, -1.0],
    [ 0.05,  1/3],
    [-0.05,  1/3],
    [-0.05, -1.0],
    [-2.00, -1.0],
    [-2.00, -1/3],
    [-2.50, -1/3],
    [-2.50,  1/3],
    [-2.00,  1/3],
    [-2.00,  1.0],
    [-1.05,  1.0],
    [-1.05, -1/3],
    [-0.95, -1/3],
    [-0.95,  1.0],
    [ 0.95,  1.0]
])

poly_path = Path(main_region)
inside_mask = poly_path.contains_points(p_fine[:, :2])
polygon_indices = np.where(inside_mask)[0]

#____________________________________________________________________________________________________________________________
# GENERATING PARAMETERS

mu_loc_indices = sample_mu_parameter_locations(
    p_coords=p_fine[:, :2],
    valid_indices=polygon_indices,
    num_samples=num_samples,
    k_params=k_params,
    min_dist=minimum_distance
) # [700, 9]

mu_locs = p_fine[mu_loc_indices, :2]  # [700, 9, 2]

mu_params = sample_mu_values(
    k_params=k_params, 
    mu_range=mu_range, 
    num_samples=num_samples
)  # [700, 9]

bases = basis_generator(p_fine[:, :2], mu_locs, power=2.0, eps=1e-8)

#____________________________________________________________________________________________________________________________
# SAVING PARAMETERS

data_dir = os.path.join(project_root, 'Data_Generation/Data')
os.makedirs(data_dir, exist_ok=True)

save_parameters_data(
    mu_values=mu_params,   # [700, 9]
    mu_loc=mu_locs,        # [700, 9, 2]
    bases=bases,           # [700, 5899, 9]
    path=data_dir,
    name='params'
)
