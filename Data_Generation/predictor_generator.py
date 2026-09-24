import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
parent_root = os.path.dirname(project_root)
for path in (project_root, parent_root):
    if path not in sys.path:
        sys.path.append(path)

import numpy as np
import torch
from scipy.stats import qmc

from matplotlib.path import Path

from data_utils import load_mesh_data

#____________________________________________________________________________________________________________________________

mesh_data_path = os.path.join(project_root, 'Data_Generation/Data', 'ex_dev.npz')
p_fine, e_fine, t_fine, p_coarse, e_coarse, t_coarse = load_mesh_data(mesh_data_path)

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

M = 9
rng = np.random.default_rng()
sample_size = min(M, len(polygon_indices))

sampler = qmc.LatinHypercube(d=9)
sample = sampler.random(n=1)
nu_parameters = qmc.scale(sample, [0.1]*9, [100]*9)

mu = np.random.uniform(low=0.1, high=100, size=9)

print(mu)