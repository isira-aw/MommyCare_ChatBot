import requests
import time
import psutil
import re
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# -----------------------------
# Custom token‑overlap F1 scorer
# -----------------------------
def f1_token_overlap(pred: str, ref: str) -> float:
    """Compute F1 based on word‑level token overlap."""
    pred_tokens = re.findall(r"\w+", pred.lower())
    ref_tokens = re.findall(r"\w+", ref.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# -----------------------------
# Semantic model
# -----------------------------
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Evaluation queries
# -----------------------------
queries = [
    ("How long does postpartum bleeding usually last?", "Postpartum bleeding, known as lochia, typically lasts up to six weeks. The bleeding is heaviest immediately after birth and gradually decreases over time. If heavy bleeding continues beyond six weeks, consult your healthcare provider."),
    ("What foods should I avoid while breastfeeding?", "Limit caffeine and avoid alcohol. Avoid fish high in mercury like shark or swordfish. Watch for any food that seems to cause discomfort to the baby."),
    ("When can I start exercising after giving birth?", "Wait until your 6-week postnatal checkup before resuming strenuous exercise. Gentle walking and pelvic floor exercises can usually start earlier."),
    ("How often should I feed my newborn?", "Feed your newborn every 2 to 3 hours, about 8 to 12 times a day. Look for hunger cues and feed on demand."),
    ("What are common signs of postpartum depression?", "Persistent sadness, low energy, anxiety, changes in sleep or appetite, feelings of guilt or worthlessness, and difficulty bonding with the baby.")
]

# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lines = text.strip().split("\n")
    filtered = [
        l for l in lines
        if not re.search(r"(sweetie|dear|congratulations|i'm here|remember|already discussed)", l, re.IGNORECASE)
    ]
    return " ".join(filtered).strip()

def is_valid(text: str) -> bool:
    return isinstance(text, str) and len(clean_text(text)) > 20

# -----------------------------
# Evaluation loop
# -----------------------------
results = []

for q, ref in queries:
    try:
        t0 = time.perf_counter()
        m0 = psutil.Process().memory_info().rss

        resp = requests.post("http://localhost:8080/get_answer/", json={"query": q})
        t1 = time.perf_counter()
        m1 = psutil.Process().memory_info().rss

        answer = resp.json().get("answer", "")
        latency_ms = (t1 - t0) * 1000
        mem_MB = (m1 - m0) / (1024 ** 2)
        tokens = len(answer.split())
        throughput = tokens / (t1 - t0) if (t1 - t0) else 0

        cleaned_ans = clean_text(answer)
        cleaned_ref = clean_text(ref)

        # Metrics
        em = 1.0 if cleaned_ans.lower().strip() == cleaned_ref.lower().strip() else 0.0
        f1_score = None
        sim_score = None

        if is_valid(cleaned_ans) and is_valid(cleaned_ref):
            f1_score = f1_token_overlap(cleaned_ans, cleaned_ref)
            try:
                ref_vec = similarity_model.encode([cleaned_ref])[0]
                ans_vec = similarity_model.encode([cleaned_ans])[0]
                sim_score = float(cosine_similarity([ref_vec], [ans_vec])[0][0])
            except Exception as e:
                print(f"⚠️ Similarity error for '{q}': {e}")

        results.append({
            "question": q,
            "answer": answer,
            "latency_ms": latency_ms,
            "mem_MB": mem_MB,
            "throughput_toks_per_s": throughput,
            "EM": em,
            "F1": f1_score,
            "Semantic_Similarity": sim_score
        })

        print(f"✅ {q} — EM: {em}, F1: {f1_score}, Sim: {sim_score}")

    except Exception as e:
        print(f"❌ Error for '{q}': {e}")
        continue

# -----------------------------
# Save results
# -----------------------------
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

# -----------------------------
# Summary stats
# -----------------------------
total = len(df)
avg_lat = df['latency_ms'].mean()
avg_mem = df['mem_MB'].mean()
avg_tp = df['throughput_toks_per_s'].mean()
avg_em = df['EM'].mean()
avg_f1 = df['F1'].dropna().mean()
avg_sim = df['Semantic_Similarity'].dropna().mean()

print("\n===== Summary Statistics =====")
print(f"Total Queries: {total}")
print(f"Avg Latency: {avg_lat:.2f} ms")
print(f"Avg Memory: {avg_mem:.2f} MB")
print(f"Avg Throughput: {avg_tp:.2f} tokens/sec")
print(f"Avg Exact Match (EM): {avg_em:.2f}")
print(f"Avg F1 Score: {avg_f1:.2f}" if not pd.isna(avg_f1) else "Avg F1 Score: N/A")
print(f"Avg Semantic Similarity: {avg_sim:.2f}" if not pd.isna(avg_sim) else "Avg Semantic Similarity: N/A")

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(df['question'], df['latency_ms'], color='skyblue')
plt.xlabel("Latency (ms)")
plt.title("Latency per Query")

plt.subplot(1, 2, 2)
plt.barh(df['question'], df['mem_MB'], color='lightgreen')
plt.xlabel("Memory (MB)")
plt.title("Memory Usage per Query")
plt.tight_layout()
plt.savefig("latency_memory_evaluation.png")
plt.show()

valid_sim = df.dropna(subset=['Semantic_Similarity'])
if not valid_sim.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(valid_sim['question'], valid_sim['Semantic_Similarity'], color='plum')
    plt.xlabel("Semantic Similarity (0–1)")
    plt.title("Semantic Similarity per Query")
    plt.tight_layout()
    plt.savefig("semantic_similarity_plot.png")
    plt.show()
