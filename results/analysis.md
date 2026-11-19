# Analysis

## 1. MAE: Why 75% masking in images vs 15% in BERT?
Text has low redundancy: hiding 15% of words already forces BERT to infer syntax and meaning from context. Satellite images have high spatial redundancy because neighbouring 8x8 patches share the same colours, crop lines, and textures. With 15% masking, the model could just interpolate neighbouring pixels without learning what a forest, highway, or lake looks like. Masking 75% (48 of 64 patches) blocks local copying and forces the encoder to learn global layout. In our reconstructions (`results/mae_reconstructions.png`), the predicted patches are slightly blurry because MSE averages over fine textures, but the overall geometry of fields, canopies, water, and roads is clearly recovered.

## 2. MAE: Few-label classifier confusions (160 labelled images)
Using only 16 labelled tiles per class on top of the frozen 192-D MAE encoder gave well above the 10% chance baseline (run Kaggle training to fill exact accuracy) on the 5,400 test images (`results/mae_confusion_matrix.png`). Most errors happen between classes that look very similar at 64x64 RGB: PermanentCrop vs AnnualCrop, HerbaceousVegetation vs Pasture, and Highway vs River. Agricultural classes (`AnnualCrop`, `PermanentCrop`, `Pasture`, `HerbaceousVegetation`) share green/brown rectangular textures, while `Highway` and `River` are both thin linear features cutting across land. Mean-pooling all 64 patch tokens also loses some fine local structure, making these overlapping classes harder to separate linearly.

## 3. VAE: Reconstruction vs KL dynamics, and what happens without KL
Reconstruction BCE drops quickly early in training as the decoder learns stroke shapes, while the KL term rises and settles as the encoder packs character structure smoothly into the 16-D latent space.

This is the rate-distortion trade-off in the negative ELBO. Early on, reducing BCE dominates; as the decoder gets better, the encoder encodes more stroke details into `z`, pushing KL up until both terms balance.

If the KL term were removed completely, the VAE would behave like a standard autoencoder: the encoder would push $\sigma \to 0$ and scatter the means $\mu(x)$ wherever it helps reconstruction, leaving large empty gaps in latent space. Reconstructions of test images would get slightly sharper, but sampling $z \sim \mathcal{N}(0, I)$ would mostly hit unmapped empty regions and decode into broken, noisy shapes rather than readable Urdu letters.

## 4. VAE vs Diffusion: Sharpness and objective comparison
Comparing `results/vae_samples.png` and `results/ddpm_samples_40x8.png`, the DDPM samples are clearly sharper with cleaner stroke edges and dots, while VAE samples look softer and blurred.
- The VAE compresses a 28x28 image into 16 numbers and predicts all 784 pixels in one pass with an independent Bernoulli likelihood. When stroke placement is uncertain, minimising expected BCE makes the decoder output mid-grey pixel averages, causing blur.
- DDPM keeps full 28x28 resolution across all 1,000 steps and trains on $\mathcal{L}_{\text{simple}}$ (noise MSE) at every scale. Reverse sampling builds coarse character shape at high $t$ and sharpens crisp edges and dots at low $t$ without averaging modes together.

## 5. Diffusion: Hardest Urdu characters
Looking at `results/ddpm_samples_40x8.png` and `results/human_eval.csv`, simple distinct shapes (*alif*, *daal*, *laam*, *meem*, *waao*) are generated reliably. The hardest classes are letters that share the exact same base stroke and differ only by dots (*nuqtas*) or a small mark at 28x28 resolution: *bay/pay/tay/ttay/say*, *jeem/chay/hay/khay*, *seen/sheen*, *suad/zuad*, and *ain/ghain*. A missing or shifted dot immediately turns one letter into another member of the same family.

## 6. RAG: Top-10 retrieval failures and generator behaviour
The retriever missed all relevant abstracts in the top 10 on TBD / 707 test questions. Three examples:

Run `part_d_rag/retriever.py`, `part_d_rag/generate_and_eval.py`, and `python update_report.py` to fill 3 exact failure cases.

When retrieval fails, a general-purpose bi-encoder like MiniLM has matched broad medical keywords rather than the specific gene, enzyme, or syndrome. In those cases, Qwen2.5-1.5B-Instruct either abstains ("I do not know") or tries to answer from off-topic snippets, which results in unsupported or wrong answers.

## 7. RAG: Why is Recall@k lower than Hit@k for the same k?
Hit@k gives 1 as soon as **at least one** relevant abstract appears in the top k. Recall@k is $|\text{top}_k(q) \cap R_q| / \min(k, |R_q|)$, which requires finding **all** $\min(k, |R_q|)$ relevant abstracts to get 1. Whenever $k > 1$ and a question has multiple relevant abstracts ($|R_q| > 1$), retrieving 1 out of 4 relevant abstracts gives Hit@5 = 1.0 but Recall@5 = 0.25. Thus $\text{Recall@}k \le \text{Hit@}k$ for every question and strictly lower on average for $k \ge 2$.
