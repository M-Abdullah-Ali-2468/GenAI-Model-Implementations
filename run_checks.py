import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from part_a_mae.model import verify_mae
from part_b_vae.model import verify_vae
from part_c_diffusion.schedule import verify_ddpm
from part_c_diffusion.model import build_unet, count_parameters as count_unet_params
from part_d_rag.retriever import verify_rag

FORBIDDEN_MODULES = {
    "timm", "sklearn", "scikit_learn", "sentence_transformers",
    "langchain", "langchain_core", "langchain_community",
    "llama_index", "faiss", "chromadb",
}
FORBIDDEN_SYMBOLS = {
    "DDPMScheduler", "DDIMScheduler", "DDPMPipeline",
    "AutoencoderKL", "ViTModel", "ViTMAEModel", "ViTMAEForPreTraining",
}


def check_forbidden_imports():
    violations, scanned = [], 0
    for py_file in sorted(ROOT.rglob("*.py")):
        rel = py_file.relative_to(ROOT)
        if any(p.startswith(".") or p == "__pycache__" for p in rel.parts) or py_file.name == "run_checks.py":
            continue
        scanned += 1
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                        violations.append(f"{rel}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
                    violations.append(f"{rel}: from {node.module} import ...")
                for alias in node.names:
                    if alias.name in FORBIDDEN_SYMBOLS:
                        violations.append(f"{rel}: from {node.module} import {alias.name}")
    assert not violations, f"Forbidden imports found: {violations}"
    return {"files_scanned": scanned, "violations": 0, "status": "PASS"}


def main():
    res_dir = ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    mae_res = verify_mae()
    vae_res = verify_vae()
    ddpm_res = verify_ddpm()
    ddpm_res["unet_parameters"] = count_unet_params(build_unet(num_classes=40))

    ckpt_dir = ROOT / "checkpoints"
    rag_res = verify_rag(
        embeddings_path=ckpt_dir / "rag_chunk_embeddings.npy",
        chunks_path=ckpt_dir / "rag_chunks.json",
    )
    audit_res = check_forbidden_imports()

    lines = [
        f"MAE:   PASS | masked={mae_res['masked_per_image']} (48/64) | visible_perturb_diff={mae_res['loss_diff_on_visible_perturbation']:.1f} | params={mae_res['total_parameters']:,}",
        f"VAE:   PASS | KL(0,0)={vae_res['kl_at_zero_mu_zero_logvar']:.6f} | min_KL_1000={vae_res['min_kl_across_1000_random_samples']:.4f} | params={vae_res['total_parameters']:,}",
        f"DDPM:  PASS | alpha_bar_T={ddpm_res['alpha_bar_T']:.6e} | x_999 mean={ddpm_res['x_999_mean']}, std={ddpm_res['x_999_std']} | unet_params={ddpm_res['unet_parameters']:,}",
        f"RAG:   PASS | max_words={rag_res['max_chunk_words']} | max_norm_err={rag_res['max_norm_error_from_1']:.2e} | perfect_retriever=1.0 | disk_index={rag_res['disk_index_verified']}",
        f"Audit: PASS | files_scanned={audit_res['files_scanned']} | forbidden_imports=0",
    ]
    report = "\n".join(lines)
    print(report)

    (res_dir / "correctness_checks.txt").write_text(report + "\n", encoding="utf-8")
    (res_dir / "correctness_checks.json").write_text(
        json.dumps({"mae": mae_res, "vae": vae_res, "ddpm": ddpm_res, "rag": rag_res, "audit": audit_res}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
