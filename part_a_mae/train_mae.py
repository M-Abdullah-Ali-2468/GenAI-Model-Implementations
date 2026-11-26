import os, json, time, sys
sys.path.extend([".", "starter"])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from eurosat import load_eurosat_split, CLASS_NAMES
from part_a_mae.model import MaskedAutoencoder, count_parameters


def save_recons(mae, x_te, y_te, device, path):
    picked, seen = [], set()
    for i, c in enumerate(y_te.tolist()):
        if c not in seen:
            seen.add(c); picked.append(i)
        if len(picked) == 8:
            break
    imgs = x_te[picked].to(device)
    noise = torch.rand(8, 64, generator=torch.Generator(device=device).manual_seed(42), device=device)
    masked, recon, orig, _ = mae.reconstruct_images(imgs, noise)

    fig, axes = plt.subplots(3, 8, figsize=(13, 5))
    for r, (name, batch) in enumerate([("Masked", masked.cpu()), ("Recon", recon.cpu()), ("Original", orig.cpu())]):
        for c in range(8):
            axes[r, c].imshow(batch[c].clamp(0, 1).permute(1, 2, 0).numpy())
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0: axes[r, c].set_title(CLASS_NAMES[int(y_te[picked[c]])], fontsize=8)
            if c == 0: axes[r, c].set_ylabel(name, fontsize=9, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)

    x_tr, _ = load_eurosat_split("train")
    x_te, y_te = load_eurosat_split("test")
    loader = DataLoader(TensorDataset(x_tr), batch_size=256, shuffle=True)

    mae = MaskedAutoencoder().to(device)
    opt = torch.optim.AdamW(mae.parameters(), lr=5e-4, weight_decay=0.05)

    losses, t0 = [], time.time()
    for ep in range(1, 101):
        mae.train()
        tot = 0.0
        for (bx,) in loader:
            bx = bx.to(device)
            opt.zero_grad()
            loss, _, _ = mae(bx)
            loss.backward(); opt.step()
            tot += loss.item() * bx.size(0)
        losses.append(tot / len(x_tr))
        if ep == 1 or ep % 10 == 0:
            print(f"MAE Epoch {ep:3d}/100 | loss={losses[-1]:.6f} | {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    n_params = count_parameters(mae)
    torch.save({"model_state_dict": mae.state_dict(), "epoch_losses": losses, "num_parameters": n_params, "train_time_sec": elapsed, "gpu": gpu}, "checkpoints/mae.pt")

    log = {"model": "MAE (encoder + decoder)", "num_parameters": n_params, "epochs": 100, "train_images": len(x_tr), "test_images": len(x_te), "epoch_losses": losses, "final_loss": losses[-1], "train_time_sec": round(elapsed, 2), "gpu": gpu}
    with open("results/mae_pretrain_log.json", "w") as f:
        json.dump(log, f, indent=2)

    plt.figure(figsize=(6.5, 4)); plt.plot(range(1, 101), losses); plt.xlabel("Epoch"); plt.ylabel("Masked MSE"); plt.title("MAE Pretraining Loss"); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("results/mae_loss.png", dpi=150); plt.close()
    save_recons(mae, x_te, y_te, device, "results/mae_reconstructions.png")


if __name__ == "__main__":
    main()
