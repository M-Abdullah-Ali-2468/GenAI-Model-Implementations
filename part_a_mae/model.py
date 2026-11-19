import sys; sys.path.extend([".", "starter"])
import torch
import torch.nn as nn
from eurosat import patchify, unpatchify, PATCH


class MaskedAutoencoder(nn.Module):
    def __init__(self, num_patches=64, patch_dim=192, enc_dim=192, dec_dim=128, mask_ratio=0.75):
        super().__init__()
        self.num_keep = int(round(num_patches * (1 - mask_ratio)))
        self.num_mask = num_patches - self.num_keep
        self.dec_dim = dec_dim

        self.patch_embed = nn.Linear(patch_dim, enc_dim)
        self.enc_pos = nn.Parameter(torch.randn(1, num_patches, enc_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(enc_dim, nhead=3, dim_feedforward=768, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=6, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.dec_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(1, num_patches, dec_dim) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(dec_dim, nhead=4, dim_feedforward=512, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2, enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, patch_dim)

    def random_masking(self, x, noise=None):
        B, L, D = x.shape
        noise = torch.rand(B, L, device=x.device) if noise is None else noise
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :self.num_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, L, device=x.device, dtype=x.dtype)
        mask[:, :self.num_keep] = 0.0
        mask = torch.gather(mask, 1, ids_restore)
        return x_vis, mask, ids_restore

    @staticmethod
    def compute_masked_loss(pred, target, mask):
        return (((pred - target) ** 2).mean(dim=-1) * mask).sum() / mask.sum()

    def forward(self, images, noise=None):
        target = patchify(images, p=PATCH)
        x = self.patch_embed(target) + self.enc_pos
        x_vis, mask, ids_restore = self.random_masking(x, noise)
        x_vis = self.enc_norm(self.encoder(x_vis))

        x_dec = self.dec_embed(x_vis)
        masks = self.mask_token.expand(images.size(0), self.num_mask, -1)
        x_full = torch.gather(torch.cat([x_dec, masks], dim=1), 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_dim))
        pred = self.dec_pred(self.dec_norm(self.decoder(x_full + self.dec_pos)))
        return self.compute_masked_loss(pred, target, mask), pred, mask

    def encode_all(self, images):
        x = self.patch_embed(patchify(images, p=PATCH)) + self.enc_pos
        return self.enc_norm(self.encoder(x)).mean(dim=1)

    @torch.no_grad()
    def reconstruct_images(self, images, noise=None, mask_fill=0.5):
        self.eval()
        target = patchify(images, p=PATCH)
        _, pred, mask = self.forward(images, noise)
        m = mask.unsqueeze(-1)
        masked = unpatchify(target * (1 - m) + mask_fill * m, p=PATCH, C=3)
        recon = unpatchify(target * (1 - m) + pred.clamp(0, 1) * m, p=PATCH, C=3)
        return masked, recon, images, mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def verify_mae():
    torch.manual_seed(0)
    mae = MaskedAutoencoder()
    x = torch.rand(8, 3, 64, 64)
    loss, pred, mask = mae(x)
    counts = mask.sum(dim=1).long().tolist()
    assert all(c == 48 for c in counts) and not torch.equal(mask[0], mask[1])

    target = patchify(x, p=PATCH)
    p_vis = pred.clone(); p_vis[mask == 0] += 1000.0
    diff_vis = abs(loss.item() - mae.compute_masked_loss(p_vis, target, mask).item())
    assert diff_vis == 0.0

    p_msk = pred.clone(); p_msk[mask == 1] += 1.0
    assert abs(loss.item() - mae.compute_masked_loss(p_msk, target, mask).item()) > 0.1
    assert mae.encode_all(x).shape == (8, 192)
    return {"masked_per_image": counts, "loss_diff_on_visible_perturbation": diff_vis, "total_parameters": count_parameters(mae), "status": "PASS"}


if __name__ == "__main__":
    print(verify_mae())
