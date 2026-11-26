import os, csv, json, sys
sys.path.extend([".", "starter"])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from viz import save_grid
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import DDPMSchedule


def save_blind_test(imgs_01, names, seed=42):
    os.makedirs("results/blind_test", exist_ok=True)
    perm = np.random.default_rng(seed).permutation(len(names)).tolist()

    key = []
    for i, c in enumerate(perm):
        arr = (imgs_01[c, 0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr, mode="L").save(f"results/blind_test/img_{i:02d}.png")
        key.append({"item_id": i, "image_file": f"results/blind_test/img_{i:02d}.png", "true_label": names[c]})

    save_grid(imgs_01[perm], "results/ddpm_blind_test_grid.png", nrow=8, titles=[f"#{i:02d}" for i in range(len(perm))])
    with open("results/blind_test/answer_key.json", "w") as f:
        json.dump({"seed": seed, "class_names": names, "items": key}, f, indent=2)

    csv_path = "results/human_eval.csv"
    prev = {int(r["item_id"]): r for r in csv.DictReader(open(csv_path))} if os.path.exists(csv_path) else {}
    cols = ["item_id", "diff_image", "diff_true_label", "diff_member1_pred", "diff_member2_pred", "rag_qid", "rag_question", "rag_supported_m1", "rag_supported_m2", "rag_correct_m1", "rag_correct_m2"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for e in key:
            p = prev.get(e["item_id"], {})
            w.writerow({**{k: p.get(k, "") for k in cols}, "item_id": e["item_id"], "diff_image": e["image_file"], "diff_true_label": e["true_label"]})


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("checkpoints/ddpm.pt", map_location=device, weights_only=False)
    names = ckpt["class_names"]

    unet = build_unet(len(names)).to(device)
    unet.load_state_dict(ckpt["model_state_dict"]); unet.eval()
    sched = DDPMSchedule(ckpt["T"], ckpt["beta_start"], ckpt["beta_end"], device)
    os.makedirs("results", exist_ok=True)

    chunks = []
    for s in range(0, len(names), 10):
        e = min(s + 10, len(names))
        y = torch.arange(s, e, device=device).repeat_interleave(8)
        imgs = (sched.p_sample_loop(unet, (len(y), 1, 28, 28), y, device) + 1) / 2
        chunks.append(imgs.cpu().view(e - s, 8, 1, 28, 28))
    grid = torch.cat(chunks)

    fig, axes = plt.subplots(len(names), 8, figsize=(9.5, 42))
    for r in range(len(names)):
        for c in range(8):
            axes[r, c].imshow(grid[r, c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(names[r], rotation=0, labelpad=34, va="center", ha="right", fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.99, bottom=0.005, wspace=0.05, hspace=0.1)
    fig.savefig("results/ddpm_samples_40x8.png", dpi=140); plt.close(fig)

    ts = [999, 750, 500, 250, 100, 0]
    _, snaps = sched.p_sample_loop(unet, (1, 1, 28, 28), torch.tensor([0], device=device), device, record_steps=ts)
    traj = torch.cat([((snaps[t].clamp(-1, 1) + 1) / 2) for t in ts])
    save_grid(traj, "results/ddpm_denoising.png", nrow=6, titles=[f"t={t}" for t in ts])

    blind = (sched.p_sample_loop(unet, (len(names), 1, 28, 28), torch.arange(len(names), device=device), device) + 1) / 2
    save_blind_test(blind.cpu(), names)
    print("Saved 40x8 grid, denoising strip, and blind test.")


if __name__ == "__main__":
    main()
