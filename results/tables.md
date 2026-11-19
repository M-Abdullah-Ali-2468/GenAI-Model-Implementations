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
