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

N_samples = 1000

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

def generate_separated_knob_samples(p_coords, valid_indices, num_samples, M=9, min_dist=0.6):
    """
    Generates unique samples of mu parameter locations, ensuring they are spatially separated.
    
    Parameters:
        p_coords      : [N_v, 2] array of spatial coordinates.
        valid_indices : 1D array of node indices that fall inside the polygon.
        num_samples   : Total number of unique dataset configurations to generate.
        M             : Number of knobs per sample.
        min_dist      : Minimum Euclidean distance enforced between any two knobs.
    """
    rng = np.random.default_rng()
    valid_coords = p_coords[valid_indices]
    N_valid = len(valid_indices)
    
    samples = []
    seen_configurations = set()
    
    while len(samples) < num_samples:

        picked_local_idx = []
        candidates = rng.permutation(N_valid)

        for cand in candidates:
            if not picked_local_idx:
                picked_local_idx.append(cand)
                continue

            dists = np.linalg.norm(valid_coords[picked_local_idx] - valid_coords[cand], axis=1)
            
            if np.all(dists >= min_dist):
                picked_local_idx.append(cand)

            if len(picked_local_idx) == M:
                break
                
        if len(picked_local_idx) == M:
            global_idx = valid_indices[picked_local_idx]
            signature = tuple(sorted(global_idx))
            
            if signature not in seen_configurations:
                seen_configurations.add(signature)
                samples.append(global_idx)

    return np.array(samples)

num_dataset_samples = 700 
knob_indices_batch = generate_separated_knob_samples(
    p_coords=p_fine[:, :2],
    valid_indices=polygon_indices,
    num_samples=num_dataset_samples,
    M=9,
    min_dist=0.6 
)

def save_parameters_data(path:str,
                            name:str='params'):
    """Saves the mu values, mu parameter locations and the basis functions into a single file in compressed ```.npz``` format."""
    
    np.savez_compressed(f'{path}/{name}.npz',
                        p_fine=p_fine,                        
                        e_fine=e_fine,
                        t_fine=t_fine,
                        p_coarse=p_coarse,                        
                        e_coarse=e_coarse,
                        t_coarse=t_coarse,)
    
    print(f"Mesh '{name}' data saved.")  

def load_parameters_data(file_path:str='Data/params.npz'):
    """Saves the mu values, mu parameter locations and the basis functions into a single file in compressed ```.npz``` format."""

    data = np.load(file_path)

    p_fine = data['p_fine']    
    e_fine = data['e_fine']
    t_fine = data['t_fine']

    p_coarse = data['p_coarse']
    e_coarse = data['e_coarse']
    t_coarse = data['t_coarse']

    return p_fine, e_fine, t_fine, p_coarse, e_coarse, t_coarse

sampler = qmc.LatinHypercube(d=9)
sample = sampler.random(n=1)
nu_parameters = qmc.scale(sample, [0.1]*9, [100]*9)

mu = np.random.uniform(low=0.1, high=100, size=9)

print(mu)