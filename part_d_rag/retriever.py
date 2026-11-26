import os, json, sys
sys.path.extend([".", "starter"])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from bioasq import load_corpus, load_questions
from part_d_rag.chunk_and_index import MiniLMEncoder, chunk_abstract, mean_pool_and_normalize


def hit_at_k(ranked, rel, k):
    rel_s = set(rel)
    return 1.0 if any(a in rel_s for a in ranked[:k]) else 0.0


def recall_at_k(ranked, rel, k):
    rel_s = set(rel)
    denom = min(k, len(rel_s))
    return sum(1 for a in ranked[:k] if a in rel_s) / denom if denom else 0.0


def mrr_at_10(ranked, rel):
    rel_s = set(rel)
    for r, a in enumerate(ranked[:10], 1):
        if a in rel_s:
            return 1.0 / r
    return 0.0


class DenseRetriever:
    def __init__(self, emb_path="checkpoints/rag_chunk_embeddings.npy", chunks_path="checkpoints/rag_chunks.json", encoder=None):
        self.embs = np.load(emb_path).astype(np.float32)
        with open(chunks_path) as f:
            self.chunks = json.load(f)
        self.encoder = encoder
        aids = np.array([c["abstract_id"] for c in self.chunks], dtype=np.int64)
        self.unique_aids, self.c2a = np.unique(aids, return_inverse=True)

    def encode_queries(self, queries):
        self.encoder = self.encoder or MiniLMEncoder()
        return self.encoder.encode(queries, batch_size=128)

    def search_chunks(self, q_emb, k=5):
        scores = self.embs @ q_emb.reshape(-1)
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in idx]

    def search_abstracts(self, q_emb, k=20):
        scores = self.embs @ q_emb.reshape(-1)
        abs_sc = np.full(len(self.unique_aids), -np.inf, dtype=np.float32)
        np.maximum.at(abs_sc, self.c2a, scores)
        idx = np.argpartition(abs_sc, -k)[-k:]
        idx = idx[np.argsort(abs_sc[idx])[::-1]]
        return [(int(self.unique_aids[int(i)]), float(abs_sc[int(i)])) for i in idx]


def verify_rag(embeddings_path="checkpoints/rag_chunk_embeddings.npy", chunks_path="checkpoints/rag_chunks.json"):
    tc = chunk_abstract(1, " ".join(f"w{i}" for i in range(410)), 150, 30)
    max_wc = max(len(c["text"].split()) for c in tc)
    assert max_wc <= 150 and tc[0]["text"].split()[-30:] == tc[1]["text"].split()[:30]

    hs, am = torch.randn(16, 12, 384), torch.ones(16, 12, dtype=torch.long)
    am[:, 8:] = 0
    norm_err = float((torch.linalg.norm(mean_pool_and_normalize(hs, am), dim=1) - 1).abs().max().item())
    assert norm_err < 1e-5

    disk_ok = False
    if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
        norm_err = float(np.abs(np.linalg.norm(np.load(embeddings_path), axis=1) - 1).max())
        with open(chunks_path) as f:
            max_wc = max(len(c["text"].split()) for c in json.load(f))
        assert norm_err < 1e-4 and max_wc <= 150
        disk_ok = True

    for rel, ranked in [([1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), (list(range(1, 16)), list(range(1, 21)))]:
        for k in (1, 5, 10):
            assert hit_at_k(ranked, rel, k) == 1.0 and recall_at_k(ranked, rel, k) == 1.0
        assert mrr_at_10(ranked, rel) == 1.0

    return {"max_chunk_words": max_wc, "max_norm_error_from_1": round(norm_err, 7), "disk_index_verified": disk_ok, "perfect_retriever_Hit_Recall_MRR": 1.0, "status": "PASS"}


def main():
    os.makedirs("results", exist_ok=True)
    corpus = load_corpus()
    qs = load_questions("test", corpus)
    ret = DenseRetriever()
    q_embs = ret.encode_queries([q["question"] for q in qs])

    hits, recalls, mrrs, fails = {1: [], 5: [], 10: []}, {k: [] for k in range(1, 21)}, [], []
    for i, q in enumerate(qs):
        top20 = ret.search_abstracts(q_embs[i], k=20)
        ranked = [a for a, _ in top20]
        rel = q["relevant_ids"]
        for k in (1, 5, 10): hits[k].append(hit_at_k(ranked, rel, k))
        for k in range(1, 21): recalls[k].append(recall_at_k(ranked, rel, k))
        mrrs.append(mrr_at_10(ranked, rel))
        if hit_at_k(ranked, rel, 10) == 0.0:
            fails.append({"qid": q["qid"], "question": q["question"], "reference_answer": q["answer"], "relevant_ids": rel, "top5_retrieved_ids": ranked[:5]})

    out = {
        "num_test_questions": len(qs),
        **{f"Hit@{k}": round(float(np.mean(hits[k])), 4) for k in (1, 5, 10)},
        **{f"Recall@{k}": round(float(np.mean(recalls[k])), 4) for k in (1, 5, 10)},
        "MRR@10": round(float(np.mean(mrrs)), 4),
        "recall_curve": {str(k): round(float(np.mean(recalls[k])), 4) for k in range(1, 21)},
        "num_failures_top10": len(fails), "sample_failures_top10": fails[:15],
    }
    with open("results/rag_retrieval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Hit@1/5/10={out['Hit@1']}/{out['Hit@5']}/{out['Hit@10']} | Recall@1/5/10={out['Recall@1']}/{out['Recall@5']}/{out['Recall@10']} | MRR@10={out['MRR@10']}")

    ks = list(range(1, 21))
    plt.figure(figsize=(7, 4)); plt.plot(ks, [np.mean(recalls[k]) for k in ks], "o-")
    plt.xticks(ks); plt.ylim(0, 1.05); plt.xlabel("k"); plt.ylabel("Recall@k"); plt.title("BioASQ Retrieval Recall@k"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results/rag_recall_curve.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
