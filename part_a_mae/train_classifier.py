import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from eurosat import load_eurosat_split, few_label_indices, CLASS_NAMES
from part_a_mae.model import MaskedAutoencoder


@torch.no_grad()
def encode_dataset(mae, imgs, device, bs=256):
    mae.eval()
    return torch.cat([mae.encode_all(imgs[i:i + bs].to(device)).cpu() for i in range(0, len(imgs), bs)])


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (ROOT / "results").mkdir(exist_ok=True)

    mae = MaskedAutoencoder().to(device)
    mae.load_state_dict(torch.load(ROOT / "checkpoints/mae.pt", map_location=device, weights_only=False)["model_state_dict"])
    mae.eval()
    for p in mae.parameters():
        p.requires_grad = False

    x_tr, y_tr = load_eurosat_split("train")
    x_te, y_te = load_eurosat_split("test")
    few = few_label_indices(y_tr, per_class=16, seed=42)

    f_few, y_few = encode_dataset(mae, x_tr[few], device).to(device), y_tr[few].to(device)
    clf = nn.Linear(192, len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, 201):
        opt.zero_grad()
        loss = loss_fn(clf(f_few), y_few)  # all 160 images in one batch
        loss.backward(); opt.step()

    f_te = encode_dataset(mae, x_te, device).to(device)
    with torch.no_grad():
        preds = clf(f_te).argmax(dim=1).cpu().numpy()

    y_np = y_te.numpy()
    acc = float((preds == y_np).mean())
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_np, preds):
        cm[t, p] += 1

    confused = sorted(
        [{"true": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], "count": int(cm[i, j])}
         for i in range(10) for j in range(10) if i != j and cm[i, j] > 0],
        key=lambda d: d["count"], reverse=True,
    )
    print(f"MAE Few-Label Test Accuracy: {acc * 100:.2f}%")

    torch.save({"classifier_state_dict": clf.state_dict(), "test_accuracy": acc, "class_names": CLASS_NAMES}, ROOT / "checkpoints/mae_classifier.pt")
    out = {"labelled_train_images": len(few), "test_images": len(y_np), "test_accuracy": round(acc, 4), "test_accuracy_pct": round(acc * 100, 2), "top_confused_pairs": confused[:10], "confusion_matrix": cm.tolist(), "class_names": CLASS_NAMES}
    (ROOT / "results/mae_classifier_results.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"MAE Few-Label Confusion Matrix (Acc = {acc * 100:.2f}%)")
    fig.tight_layout(); fig.savefig(ROOT / "results/mae_confusion_matrix.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
