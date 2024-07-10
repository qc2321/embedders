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
        device: str = "cpu",
        n_samples=16,
    ):
        super(ProductSpaceVAE, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.product_manifold = product_manifold.to(device)
        self.beta = beta
        self.device = device
        self.n_samples = n_samples

        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    def encode(
        self, x: TensorType["batch_size", "n_features"]
    ) -> (TensorType["batch_size", "n_latent"], TensorType["batch_size", "n_latent"]):
        """Must return z_mean, z_logvar"""
        return self.encoder(x)

    def decode(self, z: TensorType["batch_size", "n_latent"]) -> TensorType["batch_size", "n_features"]:
        return self.decoder(z)

    def forward(self, x: TensorType["batch_size", "n_features"]) -> TensorType["batch_size", "n_features"]:
        z_means, z_logvars = self.encode(x)
        sigma = torch.diag_embed(torch.exp(z_logvars) + 1e-8)
        z = self.product_manifold.sample(z_means, sigma)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_means, sigma

    def kl_divergence(
        self,
        z_mean: TensorType["batch_size", "n_latent"],
        sigma: TensorType["n_latent", "n_latent"],
    ) -> TensorType["batch_size"]:
        # Get KL divergence as the average of log q(z|x) - log p(z)
        # See http://joschu.net/blog/kl-approx.html for more info
        means = torch.repeat_interleave(z_mean, self.n_samples, dim=0)
        sigmas = torch.repeat_interleave(sigma, self.n_samples, dim=0)
        z_samples = self.product_manifold.sample(means, sigmas)
        log_qz = self.product_manifold.log_likelihood(z_samples, means, sigmas)
        log_pz = self.product_manifold.log_likelihood(z_samples)
        return (log_qz - log_pz).view(-1, self.n_samples).mean(dim=1)

    def elbo(self, x: TensorType["batch_size", "n_features"]) -> TensorType["batch_size"]:
        x_reconstructed, z_means, sigmas = self(x)
        kld = self.kl_divergence(z_means, sigmas)
        ll = -self.reconstruction_loss(x_reconstructed, x).sum(dim=1)
        return (ll - self.beta * kld).mean(), ll.mean(), kld.mean()
