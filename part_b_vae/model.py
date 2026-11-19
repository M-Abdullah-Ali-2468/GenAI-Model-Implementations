import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu, logvar):
        # Reparameterisation trick so gradients flow through mu and logvar
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode_logits(self, z):
        return self.dec_conv(self.fc_dec(z).view(-1, 64, 7, 7))

    def decode(self, z):
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode_logits(self.reparameterise(mu, logvar)), mu, logvar

    @staticmethod
    def kl_divergence(mu, logvar):
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()

    @staticmethod
    def elbo_loss(logits, x, mu, logvar):
        # Sum BCE over 784 pixels (nats/image), then add closed-form KL
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=(1, 2, 3)).mean()
        kl = ConvVAE.kl_divergence(mu, logvar)
        return recon + kl, recon, kl

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        return self.decode(torch.randn(n, self.latent_dim, device=device))

    @torch.no_grad()
    def interpolate(self, x1, x2, steps=10):
        self.eval()
        mu1, _ = self.encode(x1.unsqueeze(0) if x1.dim() == 3 else x1)
        mu2, _ = self.encode(x2.unsqueeze(0) if x2.dim() == 3 else x2)
        alphas = torch.linspace(0, 1, steps, device=mu1.device).unsqueeze(1)
        return self.decode((1 - alphas) * mu1 + alphas * mu2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def verify_vae():
    torch.manual_seed(0)
    vae = ConvVAE(16)
    kl0 = vae.kl_divergence(torch.zeros(32, 16), torch.zeros(32, 16)).item()
    assert abs(kl0) < 1e-7

    mu, lv = torch.randn(1000, 16) * 3, torch.randn(1000, 16) * 3
    min_kl = float((-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)).min().item())
    assert min_kl >= -1e-6
    return {"kl_at_zero_mu_zero_logvar": kl0, "min_kl_across_1000_random_samples": min_kl, "total_parameters": count_parameters(vae), "status": "PASS"}


if __name__ == "__main__":
    print(verify_vae())
