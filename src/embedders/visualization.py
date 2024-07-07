from torchtyping import TensorType
import torch


def hyperboloid_to_poincare(X: TensorType["n_points", "n_dim"]) -> TensorType["n_points", "n_dim_minus_1"]:
    """Convert hyperboloid coordinates to Poincaré ball coordinates."""
    # Spatial components: all columns except the first
    x_space = X[:, 1:]

    # Time-like component: first column, reshaped for broadcasting
    x_time = X[:, 0:1]

    # Convert to Poincaré ball coordinates
    poincare_coords = x_space / (1 + x_time)

    return poincare_coords


def spherical_to_polar(X: TensorType["n_points", "n_dim"]) -> TensorType["n_points", "n_dim_minus_1"]:
    """Convert spherical coordinates to polar coordinates."""
    # Radius computation
    r = torch.norm(X, dim=1, keepdim=True)

    # Prepare output tensor
    out = torch.zeros_like(X)
    out[:, 0] = r.squeeze()  # Set the radius

    # Compute angles
    for i in range(1, X.size(1)):
        if i == X.size(1) - 1:
            # Last angle, use atan2 for full 360 degree
            out[:, i] = torch.atan2(X[:, i - 1], X[:, i - 2])
        else:
            # Compute angle from the higher dimension 'hypotenuse'
            hypotenuse = torch.norm(X[:, i:], dim=1, keepdim=True)
            # Prevent division by zero
            safe_hypotenuse = torch.where(hypotenuse > 0, hypotenuse, torch.tensor(1.0).to(X.device))
            # Ensure acos receives values within [-1, 1] and preserve dimensions
            angle = torch.acos(torch.clamp(X[:, i : i + 1] / safe_hypotenuse, -1, 1))
            out[:, i] = angle.squeeze()

    return out[:, 1:]
