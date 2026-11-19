For our Generative AI course project, my teammate and I implemented four generative models from scratch in PyTorch and connected them in one Streamlit web app:

1. Masked Autoencoder (MAE) on EuroSAT satellite tiles: 75% random patch masking, masked-only MSE pretraining, and a frozen linear classifier trained on just 16 labelled images per class (160 total).
2. Variational Autoencoder (VAE) on UHaT handwritten Urdu characters: 16-D convolutional VAE with reparameterisation, Bernoulli decoder, and closed-form KL for sampling and latent-space morphing.
3. Class-Conditional DDPM on UHaT: custom T=1000 linear noise schedule and reverse sampling loop writing any requested Urdu character from pure noise, evaluated with a blind human test.
4. Biomedical RAG on BioASQ (40k+ PubMed abstracts): overlapping 150-word chunking, MiniLM attention-mask mean pooling, matrix-multiplication cosine search, and cited answers from Qwen2.5-1.5B-Instruct.

No high-level shortcuts (`timm`, `sentence-transformers`, `FAISS`, `LangChain`, `scikit-learn`, or `diffusers` schedulers).

Medium write-up: <INSERT_MEDIUM_LINK>
GitHub repo: <INSERT_GITHUB_LINK>

#GenerativeAI #PyTorch #DeepLearning #MAE #VAE #DiffusionModels #RAG
