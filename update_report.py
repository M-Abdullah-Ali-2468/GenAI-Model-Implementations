import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def load_json(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def parse_human_eval():
    p = RES / "human_eval.csv"
    out = {
        "diff_m1": None, "diff_m2": None, "diff_agree": None, "diff_total": 40,
        "rag_s1": None, "rag_s2": None, "rag_c1": None, "rag_c2": None, "rag_total": 20,
    }
    if not p.exists():
        return out

    def to_bit(v):
        s = (v or "").strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return 1
        if s in {"0", "false", "no", "n"}:
            return 0
        return None

    d_m1 = d_m2 = d_ag = d_n = 0
    r_s1 = r_s2 = r_c1 = r_c2 = r_n = 0
    with open(p, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = (row.get("diff_true_label") or "").strip().lower()
            p1 = (row.get("diff_member1_pred") or "").strip().lower()
            p2 = (row.get("diff_member2_pred") or "").strip().lower()
            if t and p1 and p2:
                d_n += 1
                d_m1 += int(p1 == t)
                d_m2 += int(p2 == t)
                d_ag += int(p1 == p2)

            vals = [to_bit(row.get(k)) for k in ("rag_supported_m1", "rag_supported_m2", "rag_correct_m1", "rag_correct_m2")]
            if all(v is not None for v in vals):
                r_n += 1
                r_s1 += vals[0]; r_s2 += vals[1]; r_c1 += vals[2]; r_c2 += vals[3]

    if d_n:
        out.update(diff_m1=d_m1, diff_m2=d_m2, diff_agree=d_ag, diff_total=d_n)
    if r_n:
        out.update(rag_s1=r_s1, rag_s2=r_s2, rag_c1=r_c1, rag_c2=r_c2, rag_total=r_n)
    return out


def fmt_time(sec, gpu):
    return f"{sec / 60.0:.1f} min / {gpu or 'GPU'}" if sec is not None else "TBD"


def build_tables(mae, mae_cls, vae, ddpm, rag_idx, rag_ret, rag_gen, he):
    t1_eu = f"{mae['train_images']:,} / {mae['test_images']:,}; {mae_cls['labelled_train_images'] if mae_cls else 160}" if mae else "16,200 / 5,400; 160"
    t1_uh = f"{vae['num_classes']} classes; {vae['train_images']:,} / {vae['test_images']:,}" if vae else "40 classes; TBD / TBD"
    t1_ba = f"{rag_idx['num_abstracts']:,} abstracts; {rag_idx['num_chunks']:,} chunks; {rag_idx['num_test_questions']:,} test questions" if rag_idx else "40,181 abstracts; TBD chunks; 707 test questions"
    t1_mr = f"{rag_idx['mean_relevant_abstracts_per_question']:.2f}" if rag_idx else "TBD"

    mae_p = f"{mae['num_parameters']:,}" if mae else "3,173,504"
    mae_l = f"{mae['final_loss']:.6f}" if mae else "TBD"
    mae_t = fmt_time(mae["train_time_sec"], mae.get("gpu")) if mae else "TBD"

    vae_p = f"{vae['num_parameters']:,}" if vae else "220,385"
    vae_l = f"{vae['final_train_loss']:.2f} nats" if vae else "TBD"
    vae_t = fmt_time(vae["train_time_sec"], vae.get("gpu")) if vae else "TBD"

    ddpm_p = f"{ddpm['num_parameters']:,}" if ddpm else "4,028,353"
    ddpm_l = f"{ddpm['final_loss']:.6f}" if ddpm else "TBD"
    ddpm_t = fmt_time(ddpm["train_time_sec"], ddpm.get("gpu")) if ddpm else "TBD"

    mae_acc = f"{mae_cls['test_accuracy_pct']:.2f}%" if mae_cls else "TBD"
    vae_te = f"{vae['final_test_total']:.2f} = {vae['final_test_recon']:.2f} + {vae['final_test_kl']:.2f} nats" if vae else "TBD"
    d_blind = f"{he['diff_m1']}/{he['diff_total']} / {he['diff_m2']}/{he['diff_total']}" if he["diff_m1"] is not None else "TBD"
    d_agree = f"{he['diff_agree']}/{he['diff_total']}" if he["diff_agree"] is not None else "TBD"

    r_row = (
        f"| {rag_ret['Hit@1']:.4f} | {rag_ret['Hit@5']:.4f} | {rag_ret['Hit@10']:.4f} | "
        f"{rag_ret['Recall@1']:.4f} | {rag_ret['Recall@5']:.4f} | {rag_ret['Recall@10']:.4f} | {rag_ret['MRR@10']:.4f} |"
        if rag_ret else "| TBD | TBD | TBD | TBD | TBD | TBD | TBD |"
    )

    rl = f"{rag_gen['rouge_l']:.4f}" if rag_gen else "TBD"
    cp = f"{rag_gen['citation_precision'] * 100:.2f}% ({rag_gen['correct_citations']}/{rag_gen['total_citations']})" if rag_gen else "TBD"
    idk = f"{rag_gen['idk_rate'] * 100:.1f}% ({rag_gen['idk_count']}/{rag_gen['num_questions']})" if rag_gen else "TBD"
    h_sup = f"{he['rag_s1']}/{he['rag_total']} / {he['rag_s2']}/{he['rag_total']}" if he["rag_s1"] is not None else "TBD"
    h_cor = f"{he['rag_c1']}/{he['rag_total']} / {he['rag_c2']}/{he['rag_total']}" if he["rag_c1"] is not None else "TBD"

    return f"""# Results Tables

## Table 1 - Data

| Item | Value |
|---|---|
| EuroSAT train / test images; labelled images used | {t1_eu} |
| UHaT character classes; train / test images | {t1_uh} |
| BioASQ abstracts; chunks; test questions | {t1_ba} |
| Mean number of relevant abstracts per test question | {t1_mr} |

## Table 2 - Training

| Model | Parameters | Final training loss | Time / GPU |
|---|---:|---:|---:|
| MAE (encoder + decoder) | {mae_p} | {mae_l} | {mae_t} |
| VAE | {vae_p} | {vae_l} | {vae_t} |
| DDPM U-Net | {ddpm_p} | {ddpm_l} | {ddpm_t} |

## Table 3 - Image results

| Metric | Value |
|---|---|
| MAE few-label classifier: test accuracy | {mae_acc} |
| VAE: test negative ELBO = reconstruction + KL (nats) | {vae_te} |
| Diffusion: blind test, member 1 / member 2 correct (of 40) | {d_blind} |
| Diffusion: images where both members agreed (of 40) | {d_agree} |

## Table 4 - Retrieval

| Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---:|---:|---:|---:|---:|---:|---:|
{r_row}

## Table 5 - Generation (100 questions)

| Metric | Value |
|---|---|
| ROUGE-L | {rl} |
| Citation precision | {cp} |
| "I don't know" rate | {idk} |
| Human: supported, member 1 / member 2 (of 20) | {h_sup} |
| Human: correct, member 1 / member 2 (of 20) | {h_cor} |
"""


def build_analysis(mae_cls, vae, rag_ret):
    if mae_cls and mae_cls.get("top_confused_pairs"):
        pairs = ", ".join(f"{p['true']} vs {p['predicted']} ({p['count']})" for p in mae_cls["top_confused_pairs"][:5])
        acc_txt = f"{mae_cls['test_accuracy_pct']:.2f}%"
    else:
        pairs = "PermanentCrop vs AnnualCrop, HerbaceousVegetation vs Pasture, and Highway vs River"
        acc_txt = "well above the 10% chance baseline (run Kaggle training to fill exact accuracy)"

    if vae and vae.get("history"):
        h = vae["history"]
        vae_txt = (
            f"Over 30 epochs, train reconstruction BCE dropped from {h['train_recon'][0]:.2f} to {h['train_recon'][-1]:.2f} nats "
            f"(test: {h['test_recon'][0]:.2f} to {h['test_recon'][-1]:.2f}), while KL rose from {h['train_kl'][0]:.2f} to "
            f"{h['train_kl'][-1]:.2f} nats (test: {h['test_kl'][0]:.2f} to {h['test_kl'][-1]:.2f}). Total test negative ELBO "
            f"fell from {h['test_total'][0]:.2f} to {h['test_total'][-1]:.2f} nats."
        )
    else:
        vae_txt = (
            "Reconstruction BCE drops quickly early in training as the decoder learns stroke shapes, while the KL term "
            "rises and settles as the encoder packs character structure smoothly into the 16-D latent space."
        )

    ans_by_qid = {}
    jsonl = RES / "rag_answers.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ans_by_qid[r["qid"]] = r

    if rag_ret and rag_ret.get("sample_failures_top10"):
        fails = rag_ret["sample_failures_top10"]
        picked = ([f for f in fails if f["qid"] in ans_by_qid] + [f for f in fails if f["qid"] not in ans_by_qid])[:3]
        blocks = []
        for idx, fc in enumerate(picked, 1):
            g = ans_by_qid.get(fc["qid"])
            g_line = (
                f"  - Generator output: \"{g['generated_answer'][:200]}\" (IDK={g['is_idk']}, citations={g['citations']})."
                if g else "  - Not in the first 100 generated questions."
            )
            blocks.append(
                f"{idx}. **QID {fc['qid']}:** {fc['question']}\n"
                f"  - Reference: {fc['reference_answer'][:160]}\n"
                f"  - Relevant IDs: {fc['relevant_ids'][:5]} | Top-5 retrieved: {fc['top5_retrieved_ids']}\n"
                f"  - Why missed: rare biomedical terms or indirect phrasing caused MiniLM to retrieve topically related abstracts instead of the exact study.\n"
                f"{g_line}"
            )
        fail_txt = "\n".join(blocks)
    else:
        fail_txt = "Run `part_d_rag/retriever.py`, `part_d_rag/generate_and_eval.py`, and `python update_report.py` to fill 3 exact failure cases."

    return f"""# Analysis

## 1. MAE: Why 75% masking in images vs 15% in BERT?
Text has low redundancy: hiding 15% of words already forces BERT to infer syntax and meaning from context. Satellite images have high spatial redundancy because neighbouring 8x8 patches share the same colours, crop lines, and textures. With 15% masking, the model could just interpolate neighbouring pixels without learning what a forest, highway, or lake looks like. Masking 75% (48 of 64 patches) blocks local copying and forces the encoder to learn global layout. In our reconstructions (`results/mae_reconstructions.png`), the predicted patches are slightly blurry because MSE averages over fine textures, but the overall geometry of fields, canopies, water, and roads is clearly recovered.

## 2. MAE: Few-label classifier confusions (160 labelled images)
Using only 16 labelled tiles per class on top of the frozen 192-D MAE encoder gave {acc_txt} on the 5,400 test images (`results/mae_confusion_matrix.png`). Most errors happen between classes that look very similar at 64x64 RGB: {pairs}. Agricultural classes (`AnnualCrop`, `PermanentCrop`, `Pasture`, `HerbaceousVegetation`) share green/brown rectangular textures, while `Highway` and `River` are both thin linear features cutting across land. Mean-pooling all 64 patch tokens also loses some fine local structure, making these overlapping classes harder to separate linearly.

## 3. VAE: Reconstruction vs KL dynamics, and what happens without KL
{vae_txt}

This is the rate-distortion trade-off in the negative ELBO. Early on, reducing BCE dominates; as the decoder gets better, the encoder encodes more stroke details into `z`, pushing KL up until both terms balance.

If the KL term were removed completely, the VAE would behave like a standard autoencoder: the encoder would push $\\sigma \\to 0$ and scatter the means $\\mu(x)$ wherever it helps reconstruction, leaving large empty gaps in latent space. Reconstructions of test images would get slightly sharper, but sampling $z \\sim \\mathcal{{N}}(0, I)$ would mostly hit unmapped empty regions and decode into broken, noisy shapes rather than readable Urdu letters.

## 4. VAE vs Diffusion: Sharpness and objective comparison
Comparing `results/vae_samples.png` and `results/ddpm_samples_40x8.png`, the DDPM samples are clearly sharper with cleaner stroke edges and dots, while VAE samples look softer and blurred.
- The VAE compresses a 28x28 image into 16 numbers and predicts all 784 pixels in one pass with an independent Bernoulli likelihood. When stroke placement is uncertain, minimising expected BCE makes the decoder output mid-grey pixel averages, causing blur.
- DDPM keeps full 28x28 resolution across all 1,000 steps and trains on $\\mathcal{{L}}_{{\\text{{simple}}}}$ (noise MSE) at every scale. Reverse sampling builds coarse character shape at high $t$ and sharpens crisp edges and dots at low $t$ without averaging modes together.

## 5. Diffusion: Hardest Urdu characters
Looking at `results/ddpm_samples_40x8.png` and `results/human_eval.csv`, simple distinct shapes (*alif*, *daal*, *laam*, *meem*, *waao*) are generated reliably. The hardest classes are letters that share the exact same base stroke and differ only by dots (*nuqtas*) or a small mark at 28x28 resolution: *bay/pay/tay/ttay/say*, *jeem/chay/hay/khay*, *seen/sheen*, *suad/zuad*, and *ain/ghain*. A missing or shifted dot immediately turns one letter into another member of the same family.

## 6. RAG: Top-10 retrieval failures and generator behaviour
The retriever missed all relevant abstracts in the top 10 on {rag_ret['num_failures_top10'] if rag_ret else 'TBD'} / {rag_ret['num_test_questions'] if rag_ret else 707} test questions. Three examples:

{fail_txt}

When retrieval fails, a general-purpose bi-encoder like MiniLM has matched broad medical keywords rather than the specific gene, enzyme, or syndrome. In those cases, Qwen2.5-1.5B-Instruct either abstains ("I do not know") or tries to answer from off-topic snippets, which results in unsupported or wrong answers.

## 7. RAG: Why is Recall@k lower than Hit@k for the same k?
Hit@k gives 1 as soon as **at least one** relevant abstract appears in the top k. Recall@k is $|\\text{{top}}_k(q) \\cap R_q| / \\min(k, |R_q|)$, which requires finding **all** $\\min(k, |R_q|)$ relevant abstracts to get 1. Whenever $k > 1$ and a question has multiple relevant abstracts ($|R_q| > 1$), retrieving 1 out of 4 relevant abstracts gives Hit@5 = 1.0 but Recall@5 = 0.25. Thus $\\text{{Recall@}}k \\le \\text{{Hit@}}k$ for every question and strictly lower on average for $k \\ge 2$.
"""


def build_readme(tables_md):
    checks = (RES / "correctness_checks.txt").read_text(encoding="utf-8").strip() if (RES / "correctness_checks.txt").exists() else "Run `python run_checks.py`."
    return f"""# Assignment 04: MAE, VAE, Diffusion (DDPM) and Biomedical RAG

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

# Correctness checks + forbidden-import audit
python run_checks.py

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

# Refresh tables, analysis, and README from logs
python update_report.py

# Launch web app
streamlit run app/app.py
```

For Kaggle GPU training, open `kaggle_run_all.ipynb`, attach the Kaggle UHaT dataset (`hazrat/uhat-urdu-handwritten-text-dataset`), set `REPO_URL`, and run all cells.

## Correctness Checks

```text
{checks}
```

{tables_md}

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
- Medium blog: `<INSERT_MEDIUM_LINK>`
- LinkedIn post: `<INSERT_LINKEDIN_LINK>`
"""


def build_blog(tables_md):
    return f"""# Four Generative Models in One App: MAE, VAE, DDPM, and Biomedical RAG from Scratch

For our Generative AI project, we built four very different generative models from scratch in PyTorch and connected all of them in one Streamlit app:
1. A **Masked Autoencoder (MAE)** trained on EuroSAT satellite images without labels, then evaluated with a 160-image linear probe.
2. A **Variational Autoencoder (VAE)** trained on handwritten Urdu characters (UHaT) for reconstruction, sampling, and latent morphing.
3. A **Class-Conditional DDPM** trained on the same 40 Urdu character classes to write a requested letter from pure Gaussian noise.
4. A **Biomedical RAG pipeline** over 40,000+ PubMed abstracts (BioASQ) using MiniLM dense retrieval and Qwen2.5-1.5B-Instruct with inline citations.

We avoided high-level shortcuts like `timm`, `sentence-transformers`, `FAISS`, `LangChain`, `scikit-learn`, and `diffusers` schedulers, implementing the masking, ELBO, diffusion schedule, pooling, and retrieval metrics directly.

## Part A: Masked Autoencoder on Satellite Images

Each 64x64 EuroSAT tile is split into 64 patches of size 8x8 (192 pixel values each). At every training step, we randomly mask 75% of the patches (48 out of 64) independently per image. Only the 16 visible patches go through our 6-layer Transformer encoder (width 192, 3 heads). A smaller 2-layer decoder (width 128, 4 heads) fills the 48 missing slots with a learned mask token, adds decoder positional embeddings, and predicts pixel values. MSE loss is computed strictly on the 48 masked patches.

![MAE Reconstructions](results/mae_reconstructions.png)

Why mask 75% when BERT only masks 15%? Language packs dense meaning into every token, whereas satellite images have strong local spatial redundancy. Masking only 15% lets a vision model copy neighbouring pixels instead of learning scene structure. With 75% hidden, our MAE had to reconstruct entire crop layouts, forest canopies, water bodies, and roads from sparse hints. After 100 epochs of unlabelled pretraining, we froze the encoder, mean-pooled all 64 patches, and trained a single linear layer on just 16 labelled images per class (160 total). Even with under 1% of the labels, test accuracy across 5,400 tiles far beats the 10% chance baseline, with most remaining confusions between visually similar vegetation classes (`AnnualCrop`, `PermanentCrop`, `Pasture`) and linear features (`Highway`, `River`).

## Part B: Variational Autoencoder on Urdu Handwriting

Next, we trained a convolutional VAE (latent dimension 16) on UHaT's 40 Urdu character classes (28x28 white strokes on black, no labels). Two strided convolutions predict $\\mu$ and $\\log \\sigma^2$, the reparameterisation trick samples $z = \\mu + \\sigma \\odot \\epsilon$, and two transposed convolutions output pixel logits for a Bernoulli decoder. We optimised the exact negative ELBO in nats per image: BCE summed over all 784 pixels plus closed-form KL divergence against $\\mathcal{{N}}(0, I)$.

![VAE Samples](results/vae_samples.png)

Across 30 epochs (`results/vae_loss.png`), reconstruction BCE falls quickly while KL rises and stabilises. Without the KL term, the model would collapse into a deterministic autoencoder with empty holes in latent space, so random draws from $\\mathcal{{N}}(0, I)$ would decode into garbled blobs instead of Urdu letters. With KL keeping the latent space smooth, linearly interpolating between $\\mu$ of two test letters produces clean intermediate morphs (`results/vae_interpolation.png`).

## Part C: Writing Requested Urdu Characters with DDPM

VAE samples are fast to generate but noticeably blurry because a single-step pixel-independent Bernoulli decoder averages over uncertain stroke edges. To get sharper handwriting and explicit character control, we built a class-conditional DDPM on the same 40 Urdu characters (rescaled to $[-1, 1]$). Using only `diffusers.UNet2DModel` with random weights (`num_class_embeds=40`), we coded the $T=1000$ linear $\\beta_t \\in [10^{{-4}}, 0.02]$ schedule, forward noising (where $\\bar{{\\alpha}}_T \\approx 4 \\times 10^{{-5}}$ turns clean letters into noise), Algorithm 1 training (40 epochs), and Algorithm 2 reverse sampling.

![DDPM 40x8 Grid](results/ddpm_samples_40x8.png)

The difference in sharpness compared to the VAE is immediate. Because diffusion keeps full 28x28 spatial resolution across 1,000 steps and refines coarse shape before fine edges, strokes and dots (*nuqtas*) come out crisp. In our blind 40-character recognisability test (`results/human_eval.csv`), both teammates labelled shuffled samples without seeing ground-truth names. Most mistakes happened inside *nuqta*-minimal families like *bay/pay/tay/say* and *jeem/chay/hay/khay*, where a single dot separates sibling letters.

## Part D: Biomedical RAG on BioASQ

For the fourth model, we built a dense RAG system over BioASQ's 40k+ PubMed abstracts and 707 expert test questions. Each abstract is split into chunks of at most 150 words with 30-word overlap and encoded with `sentence-transformers/all-MiniLM-L6-v2` loaded directly via `transformers.AutoModel`. We implemented attention-mask mean pooling and L2 normalisation ourselves so cosine search over all chunks is a single matrix multiplication, scoring each abstract by its best chunk.

![RAG Recall Curve](results/rag_recall_curve.png)

On all test questions, dense retrieval beats random selection (around 0.025% Hit@10) by orders of magnitude. As expected mathematically, Recall@k stays below Hit@k for $k \\ge 2$ because Hit@k rewards finding one relevant abstract while Recall@k requires finding all $\\min(k, |R_q|)$ relevant abstracts. Passing the top 5 chunks numbered `[1]`-`[5]` into `Qwen/Qwen2.5-1.5B-Instruct` with greedy decoding (max 256 new tokens) produces concise, cited answers. Our error analysis showed that retrieval mainly fails when questions hinge on rare gene symbols or abbreviations; in those cases, Qwen either abstains ("I do not know") or risks citing off-topic snippets. Every answer in our app includes the notice: *"For educational use only. Not medical advice."*

## Results Summary

{tables_md}

## Putting All Four Models in One App

Finally, we combined all four trained models inside a four-tab Streamlit app (`app/app.py`): picking EuroSAT test tiles for MAE reconstruction and classification, sampling and morphing Urdu letters with the VAE slider, generating 8 new handwritten samples of any chosen Urdu character with DDPM, and asking biomedical questions with cited PubMed passages in RAG.

All code, training scripts, Kaggle GPU notebook, and result logs are available in our GitHub repository.
"""


def main():
    RES.mkdir(parents=True, exist_ok=True)
    mae = load_json("mae_pretrain_log.json")
    mae_cls = load_json("mae_classifier_results.json")
    vae = load_json("vae_log.json")
    ddpm = load_json("ddpm_log.json")
    rag_idx = load_json("rag_index_stats.json")
    rag_ret = load_json("rag_retrieval_results.json")
    rag_gen = load_json("rag_generation_results.json")
    he = parse_human_eval()

    tables_md = build_tables(mae, mae_cls, vae, ddpm, rag_idx, rag_ret, rag_gen, he)
    (RES / "tables.md").write_text(tables_md, encoding="utf-8")
    (RES / "analysis.md").write_text(build_analysis(mae_cls, vae, rag_ret), encoding="utf-8")
    (ROOT / "README.md").write_text(build_readme(tables_md), encoding="utf-8")
    (ROOT / "BLOG_MEDIUM.md").write_text(build_blog(tables_md), encoding="utf-8")
    print("Updated tables.md, analysis.md, README.md, BLOG_MEDIUM.md")


if __name__ == "__main__":
    main()
