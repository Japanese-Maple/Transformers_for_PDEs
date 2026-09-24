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

from data_utils import (
  X_data_matrix, 
  normalize_feature_matrix
)

#____________________________________________________________________________________________________________________________

class StokesDataset(Dataset):

  def __init__(self, p, flag, mu_locs, mu_params, bases, mu_range, k_params):

    self.p = p                  # [N_v, 2]
    self.flag = flag            # [N_v]

    self.mu_locs = mu_locs      # [N_samples, k, 2]
    self.mu_params = mu_params  # [N_samples, k]
    self.bases = bases          # [N_samples, N_v, k]

    self.k = k_params
    self.mu_lim = mu_range

  def __len__(self):
    return len(self.mu_params)

  def __getitem__(self, idx):

    X_raw = X_data_matrix(
        p=self.p,
        flag=self.flag,
        mu_param=self.mu_params[idx],
        mu_loc=self.mu_locs[idx],
        bases_at_x=self.bases[idx],
    )

    return normalize_feature_matrix(
        X_raw, k_params=self.k, mu_range=self.mu_lim
    )