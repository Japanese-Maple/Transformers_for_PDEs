"""
                ┌─────────────────────────────────────────────────────────┐
                │              Column Slice Layout [5899, 39]             │
                └─────────────────────────────────────────────────────────┘
 Cols 0..1    Cols 2..10       Cols 11..19       Cols 20..28       Cols 29..37    Col 38
┌───────────┬───────────────┬─────────────────┬─────────────────┬───────────────┬────────┐
│  x , y    │ μ₁  ...  μ₉   │ p₁ˣ  ...  p₉ˣ   │ p₁ʸ  ...  p₉ʸ   │ w₁  ...  w₉   │  flag  │
└───────────┴───────────────┴─────────────────┴─────────────────┴───────────────┴────────┘
  Spatial       Viscosity       Knob X-Coords     Knob Y-Coords      RBF Weights   Boundary
   Coords      (broadcast)       (broadcast)       (broadcast)       (node-wise)     Flag
  (2 cols)      (9 cols)          (9 cols)          (9 cols)          (9 cols)     (1 col)

"""
#____________________________________________________________________________________________________________________________
from torch.utils.data import Dataset
import numpy as np
import torch

from .data_utils import (
    X_data_matrix, 
    normalize_feature_matrix
)

#____________________________________________________________________________________________________________________________

class StokesDataset(Dataset):
    """
    PyTorch Dataset for 2D Stokes flow feature matrix generation.
    Outputs normalized feature matrix X of shape [N_v, 39] per sample.
    """
    def __init__(self, p, flag, mu_locs, mu_params, bases, mu_range, k_params=9):
        # Ensure float32 precision for all input arrays
        self.p = np.asarray(p, dtype=np.float32)                   # [N_v, 2]
        self.flag = np.asarray(flag, dtype=np.float32)             # [N_v]
        
        self.mu_locs = np.asarray(mu_locs, dtype=np.float32)       # [N_samples, k, 2]
        self.mu_params = np.asarray(mu_params, dtype=np.float32)   # [N_samples, k]
        self.bases = np.asarray(bases, dtype=np.float32)           # [N_samples, N_v, k]

        self.k = k_params
        self.mu_lim = mu_range

    def __len__(self):
        return len(self.mu_params)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Construct raw 39-feature matrix [N_v, 39]
        X_raw = X_data_matrix(
            p=self.p,
            flag=self.flag,
            mu_param=self.mu_params[idx],
            mu_loc=self.mu_locs[idx],
            bases_at_x=self.bases[idx],
        )

        # Return normalized PyTorch Tensor [N_v, 39]
        return normalize_feature_matrix(
            X_raw, 
            k_params=self.k, 
            mu_range=self.mu_lim
        )