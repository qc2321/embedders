from torchtyping import TensorType

import torch

from .manifolds import ProductManifold


class ProductSpaceVAE(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        product_manifold: ProductManifold,
        beta: float = 1.0,
        reconstruction_loss: str = "mse",
    ):
        super(ProductSpaceVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.product_manifold = product_manifold
        self.beta = beta

        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    def encode(self, x: TensorType["batch_size", "n_features"]) -> TensorType["batch_size", "n_latent"]:
        return self.encoder(x)

    def decode(self, z: TensorType["batch_size", "n_latent"]) -> TensorType["batch_size", "n_features"]:
        return self.decoder(z)

    def forward(self, x: TensorType["batch_size", "n_features"]) -> TensorType["batch_size", "n_features"]:
        z_means = self.encode(x)
        z = self.product_manifold.sample(z_means)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_means

    def elbo(self, x: TensorType["batch_size", "n_features"]) -> TensorType["batch_size"]:
        x_reconstructed, z_means = self(x)
        kld = self.product_manifold.kl_divergence(z_means).sum(dim=1)
        ll = self.reconstruction_loss(x_reconstructed, x).sum(dim=1)
        return ll - self.beta * kld
