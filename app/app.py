import os, sys
sys.path.extend([".", "starter"])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from eurosat import CLASS_NAMES as EUROSAT_CLASSES, load_eurosat_split
from uhat import load_uhat_characters
from part_a_mae.model import MaskedAutoencoder
from part_b_vae.model import ConvVAE
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import DDPMSchedule
from part_d_rag.chunk_and_index import MiniLMEncoder
from part_d_rag.retriever import DenseRetriever
from part_d_rag.generate_and_eval import MEDICAL_NOTICE, QwenRAGGenerator, extract_citations

HF_REPO = "subhan1501/multimodal-generative-suite"
UHAT_PATH = "/kaggle/input/datasets/hazrat/uhat-urdu-handwritten-text-dataset" if os.path.exists("/kaggle/input/datasets/hazrat/uhat-urdu-handwritten-text-dataset") else "starter/data/uhat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Multimodal Generative Suite", layout="wide")


def get_ckpt(filename):
    path = os.path.join("checkpoints", filename)
    if os.path.exists(path):
        return path
    os.makedirs("checkpoints", exist_ok=True)
    return hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir="checkpoints")


@st.cache_resource(show_spinner="Loading MAE model...")
def load_mae():
    mae = MaskedAutoencoder().to(DEVICE)
    mae.load_state_dict(torch.load(get_ckpt("mae.pt"), map_location=DEVICE, weights_only=False)["model_state_dict"])
    clf = nn.Linear(192, 10).to(DEVICE)
    clf.load_state_dict(torch.load(get_ckpt("mae_classifier.pt"), map_location=DEVICE, weights_only=False)["classifier_state_dict"])
    return mae.eval(), clf.eval()


@st.cache_resource(show_spinner="Loading EuroSAT test split...")
def load_eurosat():
    return load_eurosat_split("test")


@st.cache_resource(show_spinner="Loading VAE model...")
def load_vae():
    ck = torch.load(get_ckpt("vae.pt"), map_location=DEVICE, weights_only=False)
    vae = ConvVAE(ck.get("latent_dim", 16)).to(DEVICE)
    vae.load_state_dict(ck["model_state_dict"])
    return vae.eval(), ck.get("class_names", [])


@st.cache_resource(show_spinner="Loading UHaT test split...")
def load_uhat_te(names_tuple):
    return load_uhat_characters(UHAT_PATH, "test", list(names_tuple) or None) if os.path.exists(UHAT_PATH) else (None, None, list(names_tuple))


@st.cache_resource(show_spinner="Loading DDPM model...")
def load_ddpm():
    ck = torch.load(get_ckpt("ddpm.pt"), map_location=DEVICE, weights_only=False)
    unet = build_unet(len(ck["class_names"])).to(DEVICE)
    unet.load_state_dict(ck["model_state_dict"])
    sched = DDPMSchedule(ck["T"], ck["beta_start"], ck["beta_end"], DEVICE)
    return unet.eval(), sched, ck["class_names"]


@st.cache_resource(show_spinner="Loading BioASQ dense index...")
def load_rag():
    return DenseRetriever(get_ckpt("rag_chunk_embeddings.npy"), get_ckpt("rag_chunks.json"), MiniLMEncoder(device=DEVICE))


@st.cache_resource(show_spinner="Loading Qwen2.5-1.5B-Instruct...")
def load_qwen():
    return QwenRAGGenerator(device=DEVICE)


st.title("Four Generative Models, One App")
st.caption(f"MAE (EuroSAT) · VAE (UHaT) · Class-Conditional DDPM (UHaT) · Biomedical RAG (BioASQ) | Weights: `{HF_REPO}` | Device: `{DEVICE}`")

t_mae, t_vae, t_ddpm, t_rag = st.tabs(["MAE (EuroSAT)", "VAE (Urdu)", "Diffusion (Urdu)", "Biomedical RAG"])

with t_mae:
    mae, clf = load_mae()
    x_te, y_te = load_eurosat()
    col_a, col_b = st.columns([3, 1])
    idx = col_a.slider("Pick EuroSAT test tile", 0, len(y_te) - 1, 0)
    st.session_state.setdefault("mae_seed", 42)
    col_b.write("")
    if col_b.button("New random mask", use_container_width=True):
        st.session_state.mae_seed += 1

    img = x_te[idx:idx + 1].to(DEVICE)
    noise = torch.rand(1, 64, generator=torch.Generator(device=DEVICE).manual_seed(st.session_state.mae_seed), device=DEVICE)
    masked, recon, orig, _ = mae.reconstruct_images(img, noise)
    with torch.no_grad():
        probs = torch.softmax(clf(mae.encode_all(img)), dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.3])
    c1.image(masked[0].cpu().permute(1, 2, 0).numpy(), caption="Masked (75%)", width=200, clamp=True)
    c2.image(recon[0].cpu().permute(1, 2, 0).numpy(), caption="Reconstruction", width=200, clamp=True)
    c3.image(orig[0].cpu().permute(1, 2, 0).numpy(), caption="Original", width=200, clamp=True)
    with c4:
        st.metric("Predicted Class", EUROSAT_CLASSES[pred], f"{probs[pred] * 100:.1f}% confidence")
        st.write(f"True class: **{EUROSAT_CLASSES[int(y_te[idx])]}**")
        for i_top in np.argsort(probs)[::-1][:3]:
            st.progress(float(probs[i_top]), text=f"{EUROSAT_CLASSES[i_top]}: {probs[i_top] * 100:.1f}%")

with t_vae:
    vae, vae_names = load_vae()
    left, right = st.columns(2)
    with left:
        st.subheader("Sample from N(0, I)")
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
        st.subheader("Latent Interpolation")
        x_u, l_u, names = load_uhat_te(tuple(vae_names))
        if x_u is not None:
            ca = st.selectbox("Character A", names, 0)
            cb = st.selectbox("Character B", names, min(15, len(names) - 1))
            ia = int((l_u == names.index(ca)).nonzero(as_tuple=True)[0][0])
            ib = int((l_u == names.index(cb)).nonzero(as_tuple=True)[0][0])
            alpha = st.slider(f"Morph ({ca} -> {cb})", 0.0, 1.0, 0.5, 0.05)
            with torch.no_grad():
                za, _ = vae.encode(x_u[ia:ia + 1].to(DEVICE))
                zb, _ = vae.encode(x_u[ib:ib + 1].to(DEVICE))
                img_a, img_b = x_u[ia, 0].numpy(), x_u[ib, 0].numpy()
                label_a, label_b = ca, cb
        else:
            st.session_state.setdefault("vae_pair_seed", 7)
            if st.button("New latent pair"):
                st.session_state.vae_pair_seed += 1
            alpha = st.slider("Interpolation alpha (A -> B)", 0.0, 1.0, 0.5, 0.05)
            torch.manual_seed(st.session_state.vae_pair_seed)
            za = torch.randn(1, vae.latent_dim, device=DEVICE)
            zb = torch.randn(1, vae.latent_dim, device=DEVICE)
            with torch.no_grad():
                img_a, img_b = vae.decode(za)[0, 0].cpu().numpy(), vae.decode(zb)[0, 0].cpu().numpy()
            label_a, label_b = "Latent A", "Latent B"

        with torch.no_grad():
            alphas = torch.linspace(0, 1, 10, device=DEVICE).unsqueeze(1)
            morph = vae.decode((1 - alpha) * za + alpha * zb)[0, 0].cpu().numpy()
            strip = vae.decode((1 - alphas) * za + alphas * zb).cpu()

        s1, s2, s3 = st.columns(3)
        s1.image(img_a, caption=label_a, width=120, clamp=True)
        s2.image(morph, caption=f"alpha={alpha:.2f}", width=120, clamp=True)
        s3.image(img_b, caption=label_b, width=120, clamp=True)

        fig2, axes2 = plt.subplots(1, 10, figsize=(10, 1.3))
        for i in range(10):
            axes2[i].imshow(strip[i, 0].numpy(), cmap="gray", vmin=0, vmax=1); axes2[i].axis("off")
        fig2.tight_layout(pad=0.1); st.pyplot(fig2); plt.close(fig2)

with t_ddpm:
    unet, sched, ddpm_names = load_ddpm()
    c1, c2 = st.columns([3, 1])
    chosen = c1.selectbox("Choose Urdu character (40 classes)", ddpm_names, 0)
    c2.write("")
    if c2.button("Generate 8 samples", type="primary", use_container_width=True):
        y = torch.full((8,), ddpm_names.index(chosen), dtype=torch.long, device=DEVICE)
        bar = st.progress(0.0, text=f"Denoising '{chosen}' (1,000 steps)...")
        imgs = ((sched.p_sample_loop(unet, (8, 1, 28, 28), y, DEVICE, progress_cb=lambda d, t: bar.progress(d / t)) + 1) / 2).cpu()
        bar.empty()
        fig, axes = plt.subplots(1, 8, figsize=(11, 1.8))
        for i, ax in enumerate(axes):
            ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1); ax.axis("off")
        fig.tight_layout(pad=0.2); st.pyplot(fig); plt.close(fig)

with t_rag:
    st.warning(MEDICAL_NOTICE)
    ret = load_rag()
    q = st.text_input("Ask a biomedical research question", placeholder="e.g., What is the role of BRCA1 in breast cancer?")
    if st.button("Answer", type="primary") and q.strip():
        with st.spinner("Retrieving relevant PubMed passages..."):
            top5 = ret.search_chunks(ret.encode_queries([q.strip()])[0], k=5)
        with st.spinner("Generating cited answer..."):
            ans = load_qwen().generate_answer(q.strip(), top5)
        cites = extract_citations(ans)
        st.info(ans)
        st.warning(MEDICAL_NOTICE)
        for rank, (ch, sc) in enumerate(top5, 1):
            tag = " [CITED]" if rank in cites else ""
            with st.expander(f"[{rank}]{tag} Abstract {ch['abstract_id']} (cos={sc:.4f})", expanded=(rank in cites or rank <= 2)):
                st.write(ch["text"])
