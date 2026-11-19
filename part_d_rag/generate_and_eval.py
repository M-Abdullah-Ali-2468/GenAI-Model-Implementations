import csv, json, re, sys, time
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from bioasq import load_corpus, load_questions
from part_d_rag.chunk_and_index import MiniLMEncoder
from part_d_rag.retriever import DenseRetriever

GENERATOR_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MEDICAL_NOTICE = "For educational use only. Not medical advice."
SYSTEM_PROMPT = (
    "Answer the biomedical question ONLY from passages [1]-[5]. Cite supporting passage numbers "
    "in square brackets like [1] or [2][4]. If the passages do not contain the answer, say "
    "\"I do not know based on the provided passages.\""
)


def extract_citations(text):
    return [int(n) for blk in re.findall(r"\[([0-9,\s]+)\]", text) for n in re.findall(r"\d+", blk) if 1 <= int(n) <= 5]


def is_idk(text):
    low = text.lower()
    return any(p in low for p in ("i do not know", "i don't know", "do not contain", "does not contain", "not mentioned in the provided", "cannot be determined"))


class QwenRAGGenerator:
    def __init__(self, model_name=GENERATOR_NAME, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def generate_answer(self, question, top_chunks, max_new_tokens=256):
        ctx = "\n\n".join(f"[{i}] (Abstract {ch['abstract_id']}) {ch['text']}" for i, (ch, _) in enumerate(top_chunks, 1))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def save_human_eval_and_samples(records, res_dir):
    picked = [records[i] for i in sorted(np.random.default_rng(42).choice(len(records), min(20, len(records)), replace=False))]
    (res_dir / "rag_human_eval_20.json").write_text(json.dumps({"seed": 42, "items": picked}, indent=2))

    csv_path = res_dir / "human_eval.csv"
    prev = {int(r["item_id"]): r for r in csv.DictReader(csv_path.open(encoding="utf-8"))} if csv_path.exists() else {}
    cols = ["item_id", "diff_image", "diff_true_label", "diff_member1_pred", "diff_member2_pred", "rag_qid", "rag_question", "rag_supported_m1", "rag_supported_m2", "rag_correct_m1", "rag_correct_m2"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i in range(max(len(prev), len(picked), 40)):
            p = prev.get(i, {})
            r = picked[i] if i < len(picked) else None
            w.writerow({**{k: p.get(k, "") for k in cols}, "item_id": i, "rag_qid": r["qid"] if r else p.get("rag_qid", ""), "rag_question": r["question"] if r else p.get("rag_question", "")})

    # 5 qualitative samples (including >=1 retrieval failure and >=1 IDK)
    sel, used = [], set()
    for tag, fn in [("Retrieval Failure", lambda r: not r["retrieval_hit_top5"]), ("Model Abstained", lambda r: r["is_idk"])]:
        for r in records:
            if fn(r) and r["qid"] not in used:
                sel.append((tag, r)); used.add(r["qid"]); break
    for r in sorted(records, key=lambda x: x["rouge_l"], reverse=True):
        if len(sel) >= 5: break
        if r["qid"] not in used:
            sel.append(("Grounded Answer", r)); used.add(r["qid"])

    md = ["# Qualitative RAG Samples", "", f"> {MEDICAL_NOTICE}", ""]
    for i, (tag, r) in enumerate(sel, 1):
        md += [f"## {i}. {tag} (QID {r['qid']})", f"**Q:** {r['question']}", f"**Answer:** {r['generated_answer']}", f"**Reference:** {r['reference_answer']}",
               f"**ROUGE-L:** {r['rouge_l']} | **Citations:** {r['citations']} | **Top-5 Hit:** {r['retrieval_hit_top5']}"]
        for k, ch in enumerate(r["retrieved_chunks"], 1):
            md.append(f"- [{k}] Abstract `{ch['abstract_id']}` (cos={ch['score']}): {ch['text'][:260]}...")
        md.append("")
    (res_dir / "samples.md").write_text("\n".join(md), encoding="utf-8")


def main():
    res_dir = ROOT / "results"; res_dir.mkdir(exist_ok=True)
    corpus = load_corpus()
    qs = load_questions("test", corpus)[:100]

    enc = MiniLMEncoder()
    ret = DenseRetriever(ROOT / "checkpoints/rag_chunk_embeddings.npy", ROOT / "checkpoints/rag_chunks.json", encoder=enc)
    q_embs = ret.encode_queries([q["question"] for q in qs])
    del enc.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen = QwenRAGGenerator()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    records, rls, tot_c, ok_c, idks, t0 = [], [], 0, 0, 0, time.time()

    for i, q in enumerate(qs):
        top5 = ret.search_chunks(q_embs[i], k=5)
        ans = gen.generate_answer(q["question"], top5)
        rel = set(q["relevant_ids"])
        cites, idk = extract_citations(ans), is_idk(ans)
        rl = float(scorer.score(q["answer"], ans)["rougeL"].fmeasure)

        tot_c += len(cites)
        ok_c += sum(1 for c in cites if top5[c - 1][0]["abstract_id"] in rel)
        idks += int(idk); rls.append(rl)

        records.append({
            "qid": q["qid"], "question": q["question"], "reference_answer": q["answer"], "relevant_ids": q["relevant_ids"],
            "retrieved_chunks": [{"rank": r, "abstract_id": ch["abstract_id"], "score": round(sc, 4), "text": ch["text"]} for r, (ch, sc) in enumerate(top5, 1)],
            "retrieval_hit_top5": any(ch["abstract_id"] in rel for ch, _ in top5),
            "generated_answer": ans, "citations": cites, "is_idk": idk, "rouge_l": round(rl, 4),
        })
        if (i + 1) % 10 == 0:
            print(f"RAG {i + 1:3d}/100 | ROUGE-L={np.mean(rls):.4f} | {time.time() - t0:.1f}s")

    (res_dir / "rag_answers.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    summary = {"num_questions": len(records), "rouge_l": round(float(np.mean(rls)), 4), "citation_precision": round(ok_c / tot_c, 4) if tot_c else 0.0, "total_citations": tot_c, "correct_citations": ok_c, "idk_rate": round(idks / len(records), 4), "idk_count": idks}
    (res_dir / "rag_generation_results.json").write_text(json.dumps(summary, indent=2))
    save_human_eval_and_samples(records, res_dir)
    print(summary)


if __name__ == "__main__":
    main()
