import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
from bioasq import load_corpus, load_questions

RETRIEVER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_abstract(abstract_id, text, max_words=150, overlap=30):
    words = text.split()
    stride = max_words - overlap
    out, start = [], 0
    while start < len(words):
        piece = words[start:start + max_words]
        out.append({"abstract_id": int(abstract_id), "text": " ".join(piece), "word_count": len(piece)})
        if start + max_words >= len(words):
            break
        start += stride
    return out


def chunk_corpus(corpus, max_words=150, overlap=30):
    chunks = []
    for aid in sorted(corpus):
        for ch in chunk_abstract(aid, corpus[aid], max_words, overlap):
            ch["chunk_id"] = len(chunks)
            chunks.append(ch)
    return chunks


def mean_pool_and_normalize(last_hidden_state, attention_mask):
    # Ignore padding tokens using attention_mask, then L2-normalise
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


class MiniLMEncoder:
    def __init__(self, model_name=RETRIEVER_NAME, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=256, max_length=256):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            embs = mean_pool_and_normalize(self.model(**enc).last_hidden_state, enc["attention_mask"])
            out.append(embs.cpu().float().numpy())
        return np.vstack(out).astype(np.float32)


def main():
    (ROOT / "checkpoints").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    t0 = time.time()
    corpus = load_corpus()
    qs = load_questions("test", corpus)
    chunks = chunk_corpus(corpus, 150, 30)
    assert max(c["word_count"] for c in chunks) <= 150

    embs = MiniLMEncoder().encode([c["text"] for c in chunks])
    norms = np.linalg.norm(embs, axis=1)
    np.save(ROOT / "checkpoints/rag_chunk_embeddings.npy", embs)
    (ROOT / "checkpoints/rag_chunks.json").write_text(json.dumps(chunks))

    stats = {
        "num_abstracts": len(corpus), "num_chunks": len(chunks),
        "num_test_questions": len(qs),
        "mean_relevant_abstracts_per_question": round(float(np.mean([len(q["relevant_ids"]) for q in qs])), 4),
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "index_time_sec": round(time.time() - t0, 2),
    }
    (ROOT / "results/rag_index_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Indexed {len(chunks):,} chunks from {len(corpus):,} abstracts in {stats['index_time_sec']}s")


if __name__ == "__main__":
    main()
