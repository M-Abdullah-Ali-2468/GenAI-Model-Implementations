import os, json, sys, time
sys.path.extend([".", "starter"])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from uhat import load_uhat_characters
from viz import save_grid
from part_c_diffusion.model import build_unet, count_parameters
from part_c_diffusion.schedule import DDPMSchedule

UHAT_PATH = "/kaggle/input/datasets/hazrat/uhat-urdu-handwritten-text-dataset" if os.path.exists("/kaggle/input/datasets/hazrat/uhat-urdu-handwritten-text-dataset") else "starter/data/uhat"


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)

    x_tr, l_tr, names = load_uhat_characters(root=UHAT_PATH, split="train")
    sched = DDPMSchedule(1000, 1e-4, 0.02)

    ts = [0, 100, 250, 500, 750, 999]
    x0 = x_tr[0:1] * 2.0 - 1.0
    eps0 = torch.randn_like(x0)
    noised = torch.cat([((x0 if t == 0 else sched.q_sample(x0, torch.tensor([t]), eps0)).clamp(-1, 1) + 1) / 2 for t in ts])
    save_grid(noised, "results/ddpm_forward_noising.png", nrow=6, titles=[f"t={t}" for t in ts])

    sched.to(device)
    loader = DataLoader(TensorDataset(x_tr * 2.0 - 1.0, l_tr), batch_size=128, shuffle=True)
    unet = build_unet(len(names)).to(device)
    opt = torch.optim.Adam(unet.parameters(), lr=2e-4)

    losses, t0 = [], time.time()
    for ep in range(1, 41):
        unet.train()
        tot = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            t = torch.randint(0, sched.T, (bx.size(0),), device=device)
            eps = torch.randn_like(bx)
            opt.zero_grad()
            loss = F.mse_loss(unet(sched.q_sample(bx, t, eps), t, class_labels=by).sample, eps)
            loss.backward(); opt.step()
            tot += loss.item() * bx.size(0)
        losses.append(tot / len(x_tr))
        if ep == 1 or ep % 5 == 0:
            print(f"DDPM Epoch {ep:2d}/40 | loss={losses[-1]:.6f} | {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    n_params = count_parameters(unet)
    torch.save({"model_state_dict": unet.state_dict(), "class_names": names, "T": 1000, "beta_start": 1e-4, "beta_end": 0.02, "num_parameters": n_params, "train_time_sec": elapsed, "gpu": gpu}, "checkpoints/ddpm.pt")

    log = {"model": "DDPM U-Net", "num_parameters": n_params, "epochs": 40, "num_classes": len(names), "class_names": names, "epoch_losses": losses, "final_loss": round(losses[-1], 6), "train_time_sec": round(elapsed, 2), "gpu": gpu}
    with open("results/ddpm_log.json", "w") as f:
        json.dump(log, f, indent=2)

    plt.figure(figsize=(6.5, 4)); plt.plot(range(1, 41), losses); plt.xlabel("Epoch"); plt.ylabel("L_simple MSE"); plt.title("DDPM Training Loss"); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("results/ddpm_loss.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
