import numpy as np
import torch

#____________________________________________________________________________________________________________________________

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