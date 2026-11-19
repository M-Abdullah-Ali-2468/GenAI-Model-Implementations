import json, sys, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from uhat import load_uhat_characters
from viz import save_grid
from part_b_vae.model import ConvVAE, count_parameters


def find_uhat():
    for p in [ROOT / "starter/data/uhat", ROOT / "data/uhat", Path("/kaggle/input/uhat-urdu-handwritten-text-dataset"), Path("/kaggle/input/datasets/hazrat/uhat-urdu-handwritten-text-dataset")]:
        if p.exists():
            return str(p)
    return str(ROOT / "starter/data/uhat")


@torch.no_grad()
def eval_vae(vae, loader, device):
    vae.eval()
    tot = rec = kl_s = n = 0.0
    for (bx,) in loader:
        bx = bx.to(device)
        logits, mu, lv = vae(bx)
        loss, recon, kl = vae.elbo_loss(logits, bx, mu, lv)
        bs = bx.size(0)
        tot += loss.item() * bs; rec += recon.item() * bs; kl_s += kl.item() * bs; n += bs
    return tot / n, rec / n, kl_s / n


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    (ROOT / "checkpoints").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)

    uhat = find_uhat()
    x_tr, _, names = load_uhat_characters(root=uhat, split="train")
    x_te, l_te, _ = load_uhat_characters(root=uhat, split="test", class_names=names)
    tr_loader = DataLoader(TensorDataset(x_tr), batch_size=128, shuffle=True)
    te_loader = DataLoader(TensorDataset(x_te), batch_size=128)

    vae = ConvVAE(16).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    hist = {k: [] for k in ("train_total", "train_recon", "train_kl", "test_total", "test_recon", "test_kl")}

    t0 = time.time()
    for ep in range(1, 31):
        vae.train()
        tot = rec = kl_s = 0.0
        for (bx,) in tr_loader:
            bx = bx.to(device)
            opt.zero_grad()
            logits, mu, lv = vae(bx)
            loss, recon, kl = vae.elbo_loss(logits, bx, mu, lv)
            loss.backward(); opt.step()
            bs = bx.size(0)
            tot += loss.item() * bs; rec += recon.item() * bs; kl_s += kl.item() * bs

        tr_t, tr_r, tr_k = tot / len(x_tr), rec / len(x_tr), kl_s / len(x_tr)
        te_t, te_r, te_k = eval_vae(vae, te_loader, device)
        for k, v in zip(hist.keys(), [tr_t, tr_r, tr_k, te_t, te_r, te_k]):
            hist[k].append(v)
        if ep == 1 or ep % 5 == 0:
            print(f"VAE Epoch {ep:2d}/30 | train={tr_t:.2f} (R={tr_r:.2f}, KL={tr_k:.2f}) | test={te_t:.2f} (R={te_r:.2f}, KL={te_k:.2f})")

    elapsed = time.time() - t0
    n_params = count_parameters(vae)
    torch.save({"model_state_dict": vae.state_dict(), "latent_dim": 16, "class_names": names, "num_parameters": n_params, "train_time_sec": elapsed, "gpu": gpu}, ROOT / "checkpoints/vae.pt")

    log = {"model": "VAE", "num_parameters": n_params, "epochs": 30, "num_classes": len(names), "class_names": names, "train_images": len(x_tr), "test_images": len(x_te), "history": hist, "final_train_loss": round(hist["train_total"][-1], 4), "final_test_total": round(hist["test_total"][-1], 4), "final_test_recon": round(hist["test_recon"][-1], 4), "final_test_kl": round(hist["test_kl"][-1], 4), "train_time_sec": round(elapsed, 2), "gpu": gpu}
    (ROOT / "results/vae_log.json").write_text(json.dumps(log, indent=2))

    # 1. Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (title, tk, ek) in zip(axes, [("Total -ELBO", "train_total", "test_total"), ("Recon BCE", "train_recon", "test_recon"), ("KL", "train_kl", "test_kl")]):
        ax.plot(range(1, 31), hist[tk], label="Train"); ax.plot(range(1, 31), hist[ek], "--", label="Test")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Nats"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "results/vae_loss.png", dpi=150); plt.close(fig)

    # 2. 16 test reconstructions
    vae.eval()
    picked, seen = [], set()
    for i, c in enumerate(l_te.tolist()):
        if c not in seen: seen.add(c); picked.append(i)
        if len(picked) == 16: break
    with torch.no_grad():
        recons = torch.sigmoid(vae(x_te[picked].to(device))[0]).cpu()
    save_grid(torch.cat([x_te[picked], recons], dim=0), str(ROOT / "results/vae_reconstructions.png"), nrow=8,
              titles=[names[int(l_te[i])] for i in picked] + ["recon"] * 16)

    # 3. 64 samples from N(0, I)
    torch.manual_seed(42)
    save_grid(vae.sample(64, device), str(ROOT / "results/vae_samples.png"), nrow=8)

    # 4. 10-step interpolation between two different characters
    i1 = 0
    c2 = (int(l_te[i1]) + len(names) // 2) % len(names)
    i2 = int((l_te == c2).nonzero(as_tuple=True)[0][0])
    interp = vae.interpolate(x_te[i1].to(device), x_te[i2].to(device), steps=10).cpu()
    save_grid(interp, str(ROOT / "results/vae_interpolation.png"), nrow=10,
              titles=[f"{names[int(l_te[i1])] if i == 0 else names[c2] if i == 9 else f'{i/9:.1f}'}" for i in range(10)])


if __name__ == "__main__":
    main()
