import numpy as np
from scipy.stats import qmc
import torch

#____________________________________________________________________________________________________________________________
# MESH DATA PROCESSING

def X_data_matrix(p:np.ndarray, flag:np.ndarray, mu_param:np.ndarray, mu_loc:np.ndarray, bases_at_x:np.ndarray) -> torch.Tensor:
    """
    Constructs the feature matrix **X** in a single pass.
    
    Parameters:
        p          : [N_v, 2] spatial coordinates
        flag       : [N_v] boundary flags
        mu_param   : [k] viscosity parameters
        mu_loc     : [k, 2] knob spatial coordinates
        bases_at_x : [N_v, k] RBF influence weights
    """
    
    Nv, k, d = p.shape[0], mu_param.shape[0], p.shape[1]
    x_data = np.zeros(shape=(Nv, 3 + 4*k), dtype=np.float32)

    x_data[:, 0     : d      ] = p
    x_data[:, d     : k+d    ] = mu_param
    x_data[:, k+d   : 3*k+d  ] = mu_loc.T.flatten()
    x_data[:, 3*k+d : 4*k+d  ] = bases_at_x
    x_data[:, 4*k+d : 4*k+d+1] = flag.reshape(-1,1)
    
    return torch.from_numpy(x_data)

def normalize_feature_matrix(X_raw: torch.Tensor, k_params: int, mu_range:list) -> torch.Tensor:
    """
    X_raw shape: [N_v, 3 + 4k] or [Batch, N_v, 3 + 4k]
    Normalizes feature groups independently so spatial features aren't drowned out.
    """
    X_scaled = X_raw.clone()
    
    idx_xy   = slice(0, 2)
    idx_mu   = slice(2, 2 + k_params)
    idx_px   = slice(2 + k_params, 2 + 2 * k_params)
    idx_py   = slice(2 + 2 * k_params, 2 + 3 * k_params)
    idx_w    = slice(2 + 3 * k_params, 2 + 4 * k_params)

    x_mean = X_raw[..., idx_xy].mean(dim=-2, keepdim=True)
    x_std  = X_raw[..., idx_xy].std(dim=-2, keepdim=True) + 1e-8
    X_scaled[..., idx_xy] = (X_raw[..., idx_xy] - x_mean) / x_std

    X_scaled[..., idx_mu] = X_raw[..., idx_mu] / (mu_range[1] - mu_range[0])

    X_scaled[..., idx_px] = X_raw[..., idx_px] / 2.5
    X_scaled[..., idx_py] = X_raw[..., idx_py] / 1.0

    w_mean = X_raw[..., idx_w].mean(dim=-2, keepdim=True)
    w_std  = X_raw[..., idx_w].std(dim=-2, keepdim=True) + 1e-8
    X_scaled[..., idx_w] = (X_raw[..., idx_w] - w_mean) / w_std

    return X_scaled

#____________________________________________________________________________________________________________________________
# PARAMETER SELECTION AND GENERATION

def sample_mu_values(k_params:int, mu_range:list, num_samples:int):
    """
    Generates unique samples of mu parameter values using LHS.
    
    Parameters:
        k_params      : Number of parameters per sample.
        mu_range      : [low, high] min and max values of mu parameters.
        num_samples   : Total number of unique dataset configurations to generate.
    """
    sampler = qmc.LatinHypercube(d=k_params)
    sample = sampler.random(n=num_samples)
    mu_samples = qmc.scale(sample, [mu_range[0]]*k_params, [mu_range[1]]*k_params)

    return mu_samples.astype(np.float32)

def sample_mu_parameter_locations(p_coords, valid_indices, num_samples, k_params=9, min_dist=0.6):
    """
    Generates unique samples of mu parameter locations, ensuring they are spatially separated.
    
    Parameters:
        p_coords      : [N_v, 2] array of spatial coordinates.
        valid_indices : 1D array of node indices that fall inside the polygon.
        num_samples   : Total number of unique dataset configurations to generate.
        k_params      : Number of parameters per sample.
        min_dist      : Minimum Euclidean distance enforced between any two parameter positions.
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

            if len(picked_local_idx) == k_params:
                break
                
        if len(picked_local_idx) == k_params:
            global_idx = valid_indices[picked_local_idx]
            signature = tuple(sorted(global_idx))
            
            if signature not in seen_configurations:
                seen_configurations.add(signature)
                samples.append(global_idx)

    return np.array(samples, dtype=np.int64)

def basis_generator(p_fine, mu_loc, power: float = 2.0, eps: float = 1e-8):
    """
    Generates the normalized Shepard interpolation basis functions given the locatios of the mu parameters.

    Parameters:
        p_fine  : Spatial node coordinates of shape [N_v, 2].
        mu_loc  : Bases center locations of shape [N_samples, k, 2].
        power   : IDW distance exponent (p=2.0 corresponds to standard inverse square weighting).
        eps     : Numerical epsilon threshold to prevent division by zero.

    Returns:
        w_basis : Normalized influence matrix of shape [N_samples, N_v, k], where row of a sample sums equal 1.0.
    """
    diff = p_fine[None, :, None, :] - mu_loc[:, None, :, :]
    d = np.linalg.norm(diff, ord=2, axis=3)

    w_basis = 1.0 / (np.where(d < eps, eps, d) ** power)
    w_basis /= w_basis.sum(axis=2, keepdims=True)

    return w_basis.astype(np.float32)
  
#____________________________________________________________________________________________________________________________
# STORING AND LOADING DATA

def save_mesh_data(p_fine, e_fine, t_fine, 
                   p_coarse, e_coarse, t_coarse,
                   b_nodes,
                   path:str,
                   name:str='ex_dev'):
    """Saves the mesh data into a single file in compressed ```.npz``` format."""
    
    np.savez_compressed(f'{path}/{name}.npz',
                        p_fine=p_fine,                        
                        e_fine=e_fine,
                        t_fine=t_fine,
                        p_coarse=p_coarse,                        
                        e_coarse=e_coarse,
                        t_coarse=t_coarse,
                        b_nodes=b_nodes)
    
    print(f"Mesh '{name}' data saved.")  

def load_mesh_data(file_path:str='Data/ex_dev.npz'):
    """Loads the mesh data from the compressed ```.npz``` format."""

    data = np.load(file_path)

    p_fine = data['p_fine']    
    e_fine = data['e_fine']
    t_fine = data['t_fine']

    p_coarse = data['p_coarse']
    e_coarse = data['e_coarse']
    t_coarse = data['t_coarse']

    b_nodes  = data['b_nodes']

    return p_fine, e_fine, t_fine, p_coarse, e_coarse, t_coarse, b_nodes

def save_parameters_data(mu_values, mu_loc, bases,
                         path:str, name:str='params'):
    """Saves the mu values, mu parameter locations and the basis functions into a single file in compressed ```.npz``` format."""
    
    np.savez_compressed(f'{path}/{name}.npz',
                        mu_values=mu_values,                        
                        mu_loc=mu_loc,
                        bases=bases
                        )
    
    print(f"Parameter '{name}' data saved.")  

def load_parameters_data(file_path:str='Data/params.npz'):
    """Saves the mu values, mu parameter locations and the basis functions into a single file in compressed ```.npz``` format."""

    data = np.load(file_path)

    mu_values = data['mu_values']    
    mu_loc    = data['mu_loc']
    bases     = data['bases']

    return mu_values, mu_loc, bases