# Assignment 04: MAE, VAE, Diffusion (DDPM) and Biomedical RAG

PyTorch implementation of four generative models—Masked Autoencoder (EuroSAT), Variational Autoencoder (UHaT), Class-Conditional DDPM (UHaT), and Biomedical RAG (BioASQ)—connected in a single 4-tab Streamlit app.

All core components are written from scratch following the allowed/forbidden library rules (no pretrained weights in Parts A-C, no `timm`, `scikit-learn`, `sentence-transformers`, `FAISS`, `LangChain`, or `diffusers` schedulers/pipelines).

## Repository Layout

```text
starter/              Given starter files (unchanged)
part_a_mae/           MAE model, pretraining, reconstructions, few-label linear classifier
part_b_vae/           ConvVAE model, training, reconstructions, samples, interpolation
part_c_diffusion/     DDPM noise schedule, U-Net constructor, training, sampling & blind test
part_d_rag/           Chunking, MiniLM index, dot-product retrieval, Qwen2.5-1.5B generation & eval
results/              Figures, tables.md, analysis.md, samples.md, rag_answers.jsonl, human_eval.csv
app/                  Streamlit 4-tab web app (app/app.py)
```

## Trained Weights

Weights are hosted externally (not committed to Git):
- Link: `<INSERT_HF_OR_GDRIVE_LINK>`
- Download and extract into `checkpoints/` (`mae.pt`, `mae_classifier.pt`, `vae.pt`, `ddpm.pt`, `rag_chunk_embeddings.npy`, `rag_chunks.json`).

## How to Run

```bash
pip install -r requirements.txt

# Task 0 (unzip UHaT into starter/data/uhat/ first)
cd starter && python check_starter.py && cd ..

# Part A
python part_a_mae/train_mae.py
python part_a_mae/train_classifier.py

# Part B
python part_b_vae/train_vae.py

# Part C
python part_c_diffusion/train_ddpm.py
python part_c_diffusion/sample_ddpm.py

# Part D
python part_d_rag/chunk_and_index.py
python part_d_rag/retriever.py
python part_d_rag/generate_and_eval.py

# Launch web app
streamlit run app/app.py
```

## Correctness Checks

```text
MAE:   PASS | masked=[48, 48, 48, 48, 48, 48, 48, 48] (48/64) | visible_perturb_diff=0.0 | params=3,173,504
VAE:   PASS | KL(0,0)=0.000000 | min_KL_1000=42.6306 | params=220,385
DDPM:  PASS | alpha_bar_T=4.035831e-05 | x_999 mean=-0.00103, std=0.99989 | unet_params=4,028,353
RAG:   PASS | max_words=150 | max_norm_err=1.00e-07 | perfect_retriever=1.0 | disk_index=False
Audit: PASS | files_scanned=24 | forbidden_imports=0
```

# Results Tables

## Table 1 - Data

| Item | Value |
|---|---|
| EuroSAT train / test images; labelled images used | 16,200 / 5,400; 160 |
| UHaT character classes; train / test images | 40 classes; TBD / TBD |
| BioASQ abstracts; chunks; test questions | 40,181 abstracts; TBD chunks; 707 test questions |
| Mean number of relevant abstracts per test question | TBD |

## Table 2 - Training

| Model | Parameters | Final training loss | Time / GPU |
|---|---:|---:|---:|
| MAE (encoder + decoder) | 3,173,504 | TBD | TBD |
| VAE | 220,385 | TBD | TBD |
| DDPM U-Net | 4,028,353 | TBD | TBD |

## Table 3 - Image results

| Metric | Value |
|---|---|
| MAE few-label classifier: test accuracy | TBD |
| VAE: test negative ELBO = reconstruction + KL (nats) | TBD |
| Diffusion: blind test, member 1 / member 2 correct (of 40) | TBD |
| Diffusion: images where both members agreed (of 40) | TBD |

## Table 4 - Retrieval

| Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---:|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 5 - Generation (100 questions)

| Metric | Value |
|---|---|
| ROUGE-L | TBD |
| Citation precision | TBD |
| "I don't know" rate | TBD |
| Human: supported, member 1 / member 2 (of 20) | TBD |
| Human: correct, member 1 / member 2 (of 20) | TBD |


## App Screenshots

| MAE Tab | VAE Tab |
|---|---|
| ![MAE](results/screenshots/tab_mae.png) | ![VAE](results/screenshots/tab_vae.png) |

| Diffusion Tab | RAG Tab |
|---|---|
| ![DDPM](results/screenshots/tab_ddpm.png) | ![RAG](results/screenshots/tab_rag.png) |

> Medical notice (Part D): *For educational use only. Not medical advice.*

## Links

- Analysis: `results/analysis.md`
- Qualitative RAG samples: `results/samples.md`
- Human evaluation sheet: `results/human_eval.csv`
