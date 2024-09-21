from torchtyping import TensorType as TT
from typing import List

import torch

from .manifolds import ProductManifold


class ProductSpaceVAE(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        pm: ProductManifold,
        beta: float = 1.0,
        reconstruction_loss: str = "mse",
        device: str = "cpu",
        n_samples=16,
    ):
        super(ProductSpaceVAE, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.pm = pm.to(device)
        self.beta = beta
        self.device = device
        self.n_samples = n_samples

        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    def encode(self, x: TT["batch_size", "n_features"]) -> (TT["batch_size", "n_latent"], TT["batch_size", "n_latent"]):
        """Must return z_mean, z_logvar"""
        return self.encoder(x)

    def decode(self, z: TT["batch_size", "n_latent"]) -> TT["batch_size", "n_features"]:
        return self.decoder(z)

    def forward(self, x: TT["batch_size", "n_features"]) -> TT["batch_size", "n_features"]:
        z_means, z_logvars = self.encode(x)
        # sigma = torch.diag_embed(torch.exp(z_logvars) + 1e-8)
        # z = self.pm.sample(z_means, sigma)
        # sigma_factorized = [torch.diag_embed(torch.exp(z_logvar) + 1e-8) for z_logvar in self.pm.factorize(z_logvars)]
        # sigma_factorized = [
        #     torch.diag_embed(torch.exp(z_logvars)[pm.intrinsic2man[i]] + 1e-8) for i in
        # ]
        sigma_factorized = self.pm.factorize(z_logvars, intrinsic=True)
        sigmas = [torch.diag_embed(torch.exp(z_logvar) + 1e-8) for z_logvar in sigma_factorized]
        z = self.pm.sample(z_means, sigmas)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_means, sigmas

    def kl_divergence(
        self,
        z_mean: TT["batch_size", "n_latent"],
        # sigma: TT["n_latent", "n_latent"],
        sigma_factorized: List[TT["batch_size", "n_latent", "n_latent"]],
    ) -> TT["batch_size"]:
        # Get KL divergence as the average of log q(z|x) - log p(z)
        # See http://joschu.net/blog/kl-approx.html for more info
        means = torch.repeat_interleave(z_mean, self.n_samples, dim=0)
        # sigmas = torch.repeat_interleave(sigma, self.n_samples, dim=0)
        sigmas_factorized_interleaved = [
            torch.repeat_interleave(sigma, self.n_samples, dim=0) for sigma in sigma_factorized
        ]
        # z_samples = self.product_manifold.sample(means, sigmas)
        z_samples = self.pm.sample(means, sigmas_factorized_interleaved)
        # log_qz = self.product_manifold.log_likelihood(z_samples, means, sigmas)
        log_qz = self.pm.log_likelihood(z_samples, means, sigmas_factorized_interleaved)
        log_pz = self.pm.log_likelihood(z_samples)
        return (log_qz - log_pz).view(-1, self.n_samples).mean(dim=1)

    def elbo(self, x: TT["batch_size", "n_features"]) -> TT["batch_size"]:
        # x_reconstructed, z_means, sigmas = self(x)
        # kld = self.kl_divergence(z_means, sigmas)
        x_reconstructed, z_means, sigma_factorized = self(x)
        kld = self.kl_divergence(z_means, sigma_factorized)
        ll = -self.reconstruction_loss(x_reconstructed, x).sum(dim=1)
        return (ll - self.beta * kld).mean(), ll.mean(), kld.mean()
