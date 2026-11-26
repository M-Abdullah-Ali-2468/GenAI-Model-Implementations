import torch


class DDPMSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device=None):
        self.T, self.beta_start, self.beta_end = T, beta_start, beta_end
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = torch.sqrt(self.alpha_bars)
        self.sqrt_omab = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_a = torch.rsqrt(self.alphas)
        self.sigmas = torch.sqrt(self.betas)
        if device is not None:
            self.to(device)

    def to(self, device):
        for k in ("betas", "alphas", "alpha_bars", "sqrt_ab", "sqrt_omab", "sqrt_recip_a", "sigmas"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return self.sqrt_ab[t].view(-1, 1, 1, 1) * x0 + self.sqrt_omab[t].view(-1, 1, 1, 1) * noise

    @torch.no_grad()
    def p_sample_loop(self, unet, shape, class_labels, device, record_steps=None, progress_cb=None):
        self.to(device); unet.eval()
        x = torch.randn(shape, device=device)
        y = class_labels.to(device=device, dtype=torch.long)
        rec_set, snaps = set(record_steps or []), {}
        if (self.T - 1) in rec_set:
            snaps[self.T - 1] = x.cpu().clone()

        for step, t in enumerate(range(self.T - 1, -1, -1)):
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = unet(x, t_b, class_labels=y).sample
            mean = self.sqrt_recip_a[t] * (x - ((1.0 - self.alphas[t]) / self.sqrt_omab[t]) * eps)
            # z = 0 at final step t = 0
            x = mean + self.sigmas[t] * torch.randn_like(x) if t > 0 else mean
            if t in rec_set and t != self.T - 1:
                snaps[t] = x.cpu().clamp(-1, 1).clone()
            if progress_cb and (step % 20 == 0 or t == 0):
                progress_cb(step + 1, self.T)

        x = x.clamp(-1.0, 1.0)
        if 0 in rec_set:
            snaps[0] = x.cpu().clone()
        return (x, snaps) if record_steps is not None else x


def verify_ddpm():
    torch.manual_seed(0)
    s = DDPMSchedule(1000, 1e-4, 0.02)
    ab_T = float(s.alpha_bars[-1].item())
    assert 1e-5 < ab_T < 1e-4
    x_T = s.q_sample(torch.ones(64, 1, 28, 28), torch.full((64,), 999, dtype=torch.long))
    m, sd = float(x_T.mean().item()), float(x_T.std().item())
    assert abs(m) < 0.05 and abs(sd - 1.0) < 0.05
    return {"alpha_bar_T": ab_T, "x_999_mean": round(m, 5), "x_999_std": round(sd, 5), "status": "PASS"}


if __name__ == "__main__":
    print(verify_ddpm())
