# Analysis

## 1. MAE: Why 75% masking in images vs 15% in BERT?
Text has low redundancy: hiding 15% of words already forces BERT to infer syntax and meaning from context. Satellite images have high spatial redundancy because neighbouring 8x8 patches share the same colours, crop lines, and textures. With 15% masking, the model could just interpolate neighbouring pixels without learning what a forest, highway, or lake looks like. Masking 75% (48 of 64 patches) blocks local copying and forces the encoder to learn global layout. In our reconstructions (`results/mae_reconstructions.png`), the predicted patches are slightly blurry because MSE averages over fine textures, but the overall geometry of fields, canopies, water, and roads is clearly recovered.

## 2. MAE: Few-label classifier confusions (160 labelled images)
Using only 16 labelled tiles per class on top of the frozen 192-D MAE encoder gave 53.72% on the 5,400 test images (`results/mae_confusion_matrix.png`). Most errors happen between classes that look very similar at 64x64 RGB: AnnualCrop vs PermanentCrop (154), SeaLake vs Pasture (149), PermanentCrop vs HerbaceousVegetation (111), River vs Highway (105), Highway vs River (97). Agricultural classes (`AnnualCrop`, `PermanentCrop`, `Pasture`, `HerbaceousVegetation`) share green/brown rectangular textures, while `Highway` and `River` are both thin linear features cutting across land. Mean-pooling all 64 patch tokens also loses some fine local structure, making these overlapping classes harder to separate linearly.

## 3. VAE: Reconstruction vs KL dynamics, and what happens without KL
Over 30 epochs, train reconstruction BCE dropped from 202.14 to 105.87 nats (test: 149.98 to 106.43), while KL rose from 14.83 to 25.03 nats (test: 16.35 to 24.71). Total test negative ELBO fell from 166.33 to 131.15 nats.

This is the rate-distortion trade-off in the negative ELBO. Early on, reducing BCE dominates; as the decoder gets better, the encoder encodes more stroke details into `z`, pushing KL up until both terms balance.

If the KL term were removed completely, the VAE would behave like a standard autoencoder: the encoder would push $\sigma \to 0$ and scatter the means $\mu(x)$ wherever it helps reconstruction, leaving large empty gaps in latent space. Reconstructions of test images would get slightly sharper, but sampling $z \sim \mathcal{N}(0, I)$ would mostly hit unmapped empty regions and decode into broken, noisy shapes rather than readable Urdu letters.

## 4. VAE vs Diffusion: Sharpness and objective comparison
Comparing `results/vae_samples.png` and `results/ddpm_samples_40x8.png`, the DDPM samples are clearly sharper with cleaner stroke edges and dots, while VAE samples look softer and blurred.
- The VAE compresses a 28x28 image into 16 numbers and predicts all 784 pixels in one pass with an independent Bernoulli likelihood. When stroke placement is uncertain, minimising expected BCE makes the decoder output mid-grey pixel averages, causing blur.
- DDPM keeps full 28x28 resolution across all 1,000 steps and trains on $\mathcal{L}_{\text{simple}}$ (noise MSE) at every scale. Reverse sampling builds coarse character shape at high $t$ and sharpens crisp edges and dots at low $t$ without averaging modes together.

## 5. Diffusion: Hardest Urdu characters
Looking at `results/ddpm_samples_40x8.png` and `results/human_eval.csv`, simple distinct shapes (*alif*, *daal*, *laam*, *meem*, *waao*) are generated reliably. The hardest classes are letters that share the exact same base stroke and differ only by dots (*nuqtas*) or a small mark at 28x28 resolution: *bay/pay/tay/ttay/say*, *jeem/chay/hay/khay*, *seen/sheen*, *suad/zuad*, and *ain/ghain*. A missing or shifted dot immediately turns one letter into another member of the same family.

## 6. RAG: Top-10 retrieval failures and generator behaviour
The retriever missed all relevant abstracts in the top 10 on 98 / 707 test questions. Three examples:

1. **QID 4213:** Is capmatinib effective for glioblastoma?
  - Reference: No. Combination of capmatinib buparlisib resulted in no clear activity in patients with recurrent PTEN-deficient glioblastoma.
  - Relevant IDs: [31776899] | Top-5 retrieved: [18366283, 18035958, 32557339, 22413865, 23363814]
  - Why missed: rare biomedical terms or indirect phrasing caused MiniLM to retrieve topically related abstracts instead of the exact study.
  - Generator output: "Yes, capmatinib is effective for glioblastoma. Passage [3] states that Capmatinib (Tabrecta™) is being developed for the treatment of glioblastoma and other cancers such as lung cancer, liver cancer, " (IDK=False, citations=[3]).
2. **QID 4456:** What is F105-P?
  - Reference: F105-P is a protamine-antibody fusion protein designed to deliver siRNA to HIV-infected or envelope-transfected cells. In specific, it was designed with the pro
  - Relevant IDs: [15908939] | Top-5 retrieved: [26058333, 24214964, 30258093, 23943237, 29059618]
  - Why missed: rare biomedical terms or indirect phrasing caused MiniLM to retrieve topically related abstracts instead of the exact study.
  - Generator output: "I do not know based on the provided passages." (IDK=True, citations=[]).
3. **QID 3807:** Which main viral protein is targeted by the drug remdesivir?
  - Reference: Viral Susceptibility to the Antiviral Remdesivir (GS-5734) Is Mediated by the Viral Polymerase and the Proofreading Exoribonuclease.
  - Relevant IDs: [30275474, 29511076] | Top-5 retrieved: [34497279, 33555378, 34535038, 23102969, 25595799]
  - Why missed: rare biomedical terms or indirect phrasing caused MiniLM to retrieve topically related abstracts instead of the exact study.
  - Generator output: "Based on the information provided in passage [1], remdesivir targets several transporters including MRP4, OATP1B1/1B3, MATE1, and OCT1. However, it does not specifically state that remdesivir targets " (IDK=True, citations=[1]).

When retrieval fails, a general-purpose bi-encoder like MiniLM has matched broad medical keywords rather than the specific gene, enzyme, or syndrome. In those cases, Qwen2.5-1.5B-Instruct either abstains ("I do not know") or tries to answer from off-topic snippets, which results in unsupported or wrong answers.

## 7. RAG: Why is Recall@k lower than Hit@k for the same k?
Hit@k gives 1 as soon as **at least one** relevant abstract appears in the top k. Recall@k is $|\text{top}_k(q) \cap R_q| / \min(k, |R_q|)$, which requires finding **all** $\min(k, |R_q|)$ relevant abstracts to get 1. Whenever $k > 1$ and a question has multiple relevant abstracts ($|R_q| > 1$), retrieving 1 out of 4 relevant abstracts gives Hit@5 = 1.0 but Recall@5 = 0.25. Thus $\text{Recall@}k \le \text{Hit@}k$ for every question and strictly lower on average for $k \ge 2$.
