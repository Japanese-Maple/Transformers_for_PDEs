"""
File prepares and sets up the domain mesh geometry for generating the Stokes flow dataset.
"""
#____________________________________________________________________________________________________________________________

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
parent_root = os.path.dirname(project_root)
for path in (project_root, parent_root):
    if path not in sys.path:
        sys.path.append(path)

import numpy as np

from Utilities.Mesh_processing import refine
from Utilities.Plot_functions import Plot_Initial_Refined_meshes
from data_utils import save_mesh_data

#____________________________________________________________________________________________________________________________

mesh_path = os.path.join(project_root, 'Meshes', 'exchanger_device_altered_mesh_data.npz')
p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(
    data_path=mesh_path,
    num_of_refinements=1,
    figsize=(21, 4),
    plot=False,
    save=False
)
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

boundary_edges = e_fine[e_fine[:, 2] > 0]
boundary_node_ids = np.unique(boundary_edges[:, 0:2]).astype(int)

if boundary_node_ids.min() >= 1:
    boundary_node_ids -= 1
Nv = p_fine.shape[0]
b_nodes = np.zeros((Nv, 1), dtype=np.float32)
b_nodes[boundary_node_ids] = 1.0

data_path = os.path.join(project_root, 'Data_Generation/Data')
save_mesh_data(
    p_fine, e_fine, t_fine,
    p_coarse, e_coarse, t_coarse,
    b_nodes,
    path=data_path, name='ex_dev')