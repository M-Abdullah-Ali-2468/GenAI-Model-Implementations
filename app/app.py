import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from eurosat import CLASS_NAMES as EUROSAT_CLASSES, load_eurosat_split
from uhat import load_uhat_characters
from part_a_mae.model import MaskedAutoencoder
from part_b_vae.model import ConvVAE
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import DDPMSchedule
from part_d_rag.chunk_and_index import MiniLMEncoder
from part_d_rag.retriever import DenseRetriever
from part_d_rag.generate_and_eval import MEDICAL_NOTICE, QwenRAGGenerator, extract_citations

CKPT = ROOT / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Four Generative Models", layout="wide")


@st.cache_resource
def load_mae():
    if not (CKPT / "mae.pt").exists() or not (CKPT / "mae_classifier.pt").exists():
        return None, None
    mae = MaskedAutoencoder().to(DEVICE)
    mae.load_state_dict(torch.load(CKPT / "mae.pt", map_location=DEVICE, weights_only=False)["model_state_dict"])
    clf = nn.Linear(192, 10).to(DEVICE)
    clf.load_state_dict(torch.load(CKPT / "mae_classifier.pt", map_location=DEVICE, weights_only=False)["classifier_state_dict"])
    return mae.eval(), clf.eval()


@st.cache_resource
def load_eurosat():
    return load_eurosat_split("test")


@st.cache_resource
def load_vae():
    if not (CKPT / "vae.pt").exists():
        return None, []
    ck = torch.load(CKPT / "vae.pt", map_location=DEVICE, weights_only=False)
    vae = ConvVAE(ck.get("latent_dim", 16)).to(DEVICE)
    vae.load_state_dict(ck["model_state_dict"])
    return vae.eval(), ck.get("class_names", [])


@st.cache_resource
def load_uhat_te(names_tuple):
    root = ROOT / "starter/data/uhat"
    return load_uhat_characters(str(root), "test", list(names_tuple) or None) if root.exists() else (None, None, list(names_tuple))


@st.cache_resource
def load_ddpm():
    if not (CKPT / "ddpm.pt").exists():
        return None, None, []
    ck = torch.load(CKPT / "ddpm.pt", map_location=DEVICE, weights_only=False)
    unet = build_unet(len(ck["class_names"])).to(DEVICE)
    unet.load_state_dict(ck["model_state_dict"])
    sched = DDPMSchedule(ck["T"], ck["beta_start"], ck["beta_end"], DEVICE)
    return unet.eval(), sched, ck["class_names"]


@st.cache_resource
def load_rag():
    if not (CKPT / "rag_chunk_embeddings.npy").exists():
        return None
    return DenseRetriever(CKPT / "rag_chunk_embeddings.npy", CKPT / "rag_chunks.json", MiniLMEncoder(device=DEVICE))


@st.cache_resource
def load_qwen():
    return QwenRAGGenerator(device=DEVICE)


st.title("Four Generative Models, One App")
t_mae, t_vae, t_ddpm, t_rag = st.tabs(["MAE (EuroSAT)", "VAE (Urdu)", "Diffusion (Urdu)", "Biomedical RAG"])

with t_mae:
    mae, clf = load_mae()
    if mae is None:
        st.warning("Train Part A first (checkpoints/mae.pt missing).")
    else:
        x_te, y_te = load_eurosat()
        idx = st.slider("Pick test tile", 0, len(y_te) - 1, 0)
        st.session_state.setdefault("mae_seed", 42)
        if st.button("New random mask"):
            st.session_state.mae_seed += 1
        img = x_te[idx:idx + 1].to(DEVICE)
        noise = torch.rand(1, 64, generator=torch.Generator(device=DEVICE).manual_seed(st.session_state.mae_seed), device=DEVICE)
        masked, recon, orig, _ = mae.reconstruct_images(img, noise)
        with torch.no_grad():
            probs = torch.softmax(clf(mae.encode_all(img)), dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

        c1, c2, c3, c4 = st.columns(4)
        c1.image(masked[0].cpu().permute(1, 2, 0).numpy(), caption="Masked (75%)", width=200, clamp=True)
        c2.image(recon[0].cpu().permute(1, 2, 0).numpy(), caption="Reconstruction", width=200, clamp=True)
        c3.image(orig[0].cpu().permute(1, 2, 0).numpy(), caption="Original", width=200, clamp=True)
        c4.metric("Predicted Class", EUROSAT_CLASSES[pred], f"{probs[pred]*100:.1f}%")
        c4.write(f"True class: **{EUROSAT_CLASSES[int(y_te[idx])]}**")

with t_vae:
    vae, vae_names = load_vae()
    if vae is None:
        st.warning("Train Part B first (checkpoints/vae.pt missing).")
    else:
        left, right = st.columns(2)
        with left:
            st.session_state.setdefault("vae_seed", 0)
            if st.button("Generate 16 random characters", type="primary"):
                st.session_state.vae_seed += 1
            torch.manual_seed(st.session_state.vae_seed)
            imgs = vae.sample(16, DEVICE).cpu()
            fig, axes = plt.subplots(4, 4, figsize=(4.5, 4.5))
            for i, ax in enumerate(axes.flat):
                ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1); ax.axis("off")
            fig.tight_layout(pad=0.2); st.pyplot(fig); plt.close(fig)

        with right:
            x_u, l_u, names = load_uhat_te(tuple(vae_names))
            if x_u is None:
                st.info("Unzip UHaT into starter/data/uhat/ to interpolate test characters.")
            else:
                ca = st.selectbox("Character A", names, 0)
                cb = st.selectbox("Character B", names, min(15, len(names) - 1))
                ia = int((l_u == names.index(ca)).nonzero(as_tuple=True)[0][0])
                ib = int((l_u == names.index(cb)).nonzero(as_tuple=True)[0][0])
                alpha = st.slider(f"Morph ({ca} -> {cb})", 0.0, 1.0, 0.5, 0.05)
                with torch.no_grad():
                    ma, _ = vae.encode(x_u[ia:ia + 1].to(DEVICE))
                    mb, _ = vae.encode(x_u[ib:ib + 1].to(DEVICE))
                    morph = vae.decode((1 - alpha) * ma + alpha * mb)[0, 0].cpu().numpy()
                s1, s2, s3 = st.columns(3)
                s1.image(x_u[ia, 0].numpy(), caption=ca, width=120, clamp=True)
                s2.image(morph, caption=f"alpha={alpha:.2f}", width=120, clamp=True)
                s3.image(x_u[ib, 0].numpy(), caption=cb, width=120, clamp=True)

with t_ddpm:
    unet, sched, ddpm_names = load_ddpm()
    if unet is None:
        st.warning("Train Part C first (checkpoints/ddpm.pt missing).")
    else:
        chosen = st.selectbox("Choose Urdu character", ddpm_names, 0)
        if st.button("Generate 8 samples", type="primary"):
            y = torch.full((8,), ddpm_names.index(chosen), dtype=torch.long, device=DEVICE)
            bar = st.progress(0.0, text=f"Sampling {chosen}...")
            imgs = ((sched.p_sample_loop(unet, (8, 1, 28, 28), y, DEVICE, progress_cb=lambda d, t: bar.progress(d / t)) + 1) / 2).cpu()
            bar.empty()
            fig, axes = plt.subplots(1, 8, figsize=(11, 1.8))
            for i, ax in enumerate(axes):
                ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1); ax.axis("off")
            fig.tight_layout(pad=0.2); st.pyplot(fig); plt.close(fig)

with t_rag:
    st.warning(MEDICAL_NOTICE)
    ret = load_rag()
    if ret is None:
        st.warning("Build RAG index first (checkpoints/rag_chunk_embeddings.npy missing).")
    else:
        q = st.text_input("Ask a biomedical question", placeholder="e.g., What is the role of BRCA1 in breast cancer?")
        if st.button("Answer", type="primary") and q.strip():
            top5 = ret.search_chunks(ret.encode_queries([q.strip()])[0], k=5)
            ans = load_qwen().generate_answer(q.strip(), top5)
            cites = extract_citations(ans)
            st.info(ans)
            st.warning(MEDICAL_NOTICE)
            for rank, (ch, sc) in enumerate(top5, 1):
                tag = " [CITED]" if rank in cites else ""
                with st.expander(f"[{rank}]{tag} Abstract {ch['abstract_id']} (cos={sc:.4f})", expanded=(rank in cites or rank <= 2)):
                    st.write(ch["text"])
