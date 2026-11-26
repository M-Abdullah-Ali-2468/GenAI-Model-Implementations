# Results Tables

## Table 1 - Data

| Item | Value |
|---|---|
| EuroSAT train / test images; labelled images used | 16,200 / 5,400; 160 |
| UHaT character classes; train / test images | 40 classes; 28,328 / 4,747 |
| BioASQ abstracts; chunks; test questions | 40,181 abstracts; 79,332 chunks; 707 test questions |
| Mean number of relevant abstracts per test question | 8.66 |

## Table 2 - Training

| Model | Parameters | Final training loss | Time / GPU |
|---|---:|---:|---:|
| MAE (encoder + decoder) | 3,173,504 | 0.006059 | 8.0 min / Tesla T4 |
| VAE | 220,385 | 130.90 nats | 0.6 min / Tesla T4 |
| DDPM U-Net | 4,028,353 | 0.022605 | 34.7 min / Tesla T4 |

## Table 3 - Image results

| Metric | Value |
|---|---|
| MAE few-label classifier: test accuracy | 53.72% |
| VAE: test negative ELBO = reconstruction + KL (nats) | 131.15 = 106.43 + 24.71 nats |
| Diffusion: blind test, member 1 / member 2 correct (of 40) | TBD |
| Diffusion: images where both members agreed (of 40) | TBD |

## Table 4 - Retrieval

| Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6223 | 0.8161 | 0.8614 | 0.6223 | 0.5863 | 0.6024 | 0.7061 |

## Table 5 - Generation (100 questions)

| Metric | Value |
|---|---|
| ROUGE-L | 0.2169 |
| Citation precision | 72.31% (47/65) |
| "I don't know" rate | 15.0% (15/100) |
| Human: supported, member 1 / member 2 (of 20) | TBD |
| Human: correct, member 1 / member 2 (of 20) | TBD |
