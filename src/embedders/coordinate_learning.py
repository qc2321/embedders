import sys

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
import numpy as np
import geoopt

from torchtyping import TensorType

from .metrics import distortion_loss, d_avg
from .manifolds import ProductManifold


def _resample(x_embed, nan_rows, pm, dims, man2dim, device):
    raise NotImplementedError


def train_coords(
    pm: ProductManifold,
    dists: TensorType["n_points", "n_points"],
    device: str = "cpu",
    burn_in_learning_rate: float = 1e-3,
    burn_in_iterations: int = 2_000,
    learning_rate: float = 1e-2,
    scale_factor_learning_rate: float = 0.0,  # Off by default
    training_iterations: int = 18_000,
    loss_window_size: int = 100,
):
    # Move everything to the device
    pm.x_embed = pm.initialize_embeddings(n_points=len(dists)).to(device)
    dists = dists.to(device)

    # Initialize optimizer
    pm.x_embed = geoopt.ManifoldParameter(pm.x_embed, manifold=pm.manifold)
    pm.opt = geoopt.optim.RiemannianAdam(
        [
            {"params": [pm.x_embed], "lr": burn_in_learning_rate},
            {"params": [x._log_scale for x in pm.manifold.manifolds], "lr": 0},
        ]
    )

    # Init TQDM
    my_tqdm = tqdm(total=burn_in_iterations + training_iterations)

    # Outer training loop - mostly setting optimizer learning rates up here
    pm.losses = []
    for lr, n_iters in ((burn_in_learning_rate, burn_in_iterations), (learning_rate, training_iterations)):
        # Set the learning rate
        pm.opt.param_groups[0]["lr"] = lr
        if lr == learning_rate:
            pm.opt.param_groups[1]["lr"] = scale_factor_learning_rate

        # Actual training loop
        for i in range(n_iters):
            # Complicated NaN resampling logic
            with torch.no_grad():
                nan_rows = torch.where(torch.isnan(pm.x_embed))[0]
                if len(nan_rows) > 0:
                    break
                    _resample(x_embed, nan_rows, pm, dims, man2dim, device)

            pm.opt.zero_grad()
            dist_est = pm.pdist(pm.x_embed)
            L = distortion_loss(dist_est, dists)
            L.backward()
            pm.losses.append(L.item())
            pm.opt.step()

            # TQDM management
            my_tqdm.update(1)
            my_tqdm.set_description(f"Loss: {L.item():.3e}, Average Loss: {np.mean(pm.losses[-loss_window_size:]):.3e}")

            d = {f"r{i}": f"{x._log_scale.item():.3f}" for i, x in enumerate(pm.manifold.manifolds)}
            d_avg_this_iter = d_avg(dist_est, dists)
            d["d_avg"] = f"{d_avg_this_iter:.4f}"
            my_tqdm.set_postfix(d)
