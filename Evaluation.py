import requests
import time
import psutil
import re
import json
import matplotlib.pyplot as plt
from evaluate import load
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load evaluation metrics
em_metric = load("exact_match")
bleu_metric = load("bleu")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Manual F1 function
def compute_f1(prediction, reference):
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Clean unwanted terms
def clean_text(text):
    if not isinstance(text, str):
        return ""
    lines = text.strip().split("\n")
    lines = [l for l in lines if not re.search(r"(sweetie|dear|congratulations|i'm here|remember|already discussed)", l, re.IGNORECASE)]
    return " ".join(lines).strip().lower()

# Check if cleaned text is valid
def is_valid_text(text):
    return isinstance(text, str) and len(clean_text(text)) > 20

# Input queries and references
queries = [
     ("What are some good postpartum exercises?", "Gentle exercises like walking, pelvic floor exercises, and stretching are beneficial postpartum. Always consult your healthcare provider before starting any exercise routine. [Source: NHS]"),  
    ("What foods should I avoid while breastfeeding?", "Limit caffeine and avoid alcohol. Avoid fish high in mercury like shark or swordfish. Watch for any food that seems to cause discomfort to the baby."),
    ("When can I start exercising after giving birth?", "Wait until your 6-week postnatal checkup before resuming strenuous exercise. Gentle walking and pelvic floor exercises can usually start earlier."),
    ("How often should I feed my newborn?", "Feed your newborn every 2 to 3 hours, about 8 to 12 times a day. Look for hunger cues and feed on demand."),
    ("What are common signs of postpartum depression?", "Persistent sadness, low energy, anxiety, changes in sleep or appetite, feelings of guilt or worthlessness, and difficulty bonding with the baby."),
    ("How do I know if my baby is getting enough milk?", "Signs include regular weight gain, feeding 8 to 12 times in 24 hours, hearing swallowing sounds during feeds, and having at least six wet diapers and three bowel movements daily by the fifth day. Your baby should also appear content after feeding. [Source: CDC]"),
    ("What is the best way to soothe a colicky baby?", "Effective methods include holding or cuddling your baby, gentle rocking, using a pacifier, swaddling, giving a warm bath, and providing white noise. It's also helpful to ensure your baby is not hungry and to burp them after feeds. [Source: Mayo Clinic]"),
    ("When should I schedule my baby's first doctor appointment?", "The first pediatrician visit should occur within 3 to 5 days after birth. This checkup assesses your baby's weight, feeding, and overall health. [Source: HealthPartners]"),
    ("Is it normal to feel overwhelmed after childbirth?", "Yes, experiencing a range of emotions, including feeling overwhelmed, is common due to hormonal changes, sleep deprivation, and the responsibilities of caring for a newborn. If these feelings persist or intensify, consult your healthcare provider. [Source: NHS]"),
    ("What are baby blues and how are they different from postpartum depression?", "Baby blues involve mood swings, crying spells, and anxiety, typically resolving within two weeks postpartum. Postpartum depression is more severe, with symptoms lasting longer and potentially interfering with daily life. If symptoms persist beyond two weeks, seek medical advice. [Source: NHS]"),
    ("How can I increase my breast milk supply?", "To increase breast milk supply, ensure frequent breastfeeding or pumping, maintain a healthy diet, stay hydrated, and get adequate rest. Consulting a lactation consultant can also provide personalized strategies. [Source: La Leche League International]"),
    ("How much sleep does a newborn need?", "Newborns typically sleep about 16 to 17 hours per day, divided into short periods of 2 to 4 hours. Their sleep patterns are irregular in the first few weeks. [Source: American Academy of Pediatrics]"),
    ("Can I take painkillers while breastfeeding?", "Some painkillers, like acetaminophen and ibuprofen, are generally considered safe during breastfeeding. However, always consult your healthcare provider before taking any medication. [Source: NHS]"),
    ("What are signs of a good latch during breastfeeding?", "Signs include baby's mouth covering a large portion of the areola, rhythmic sucking and swallowing, and absence of pain for the mother. The baby's lips should be flanged outward. [Source: La Leche League International]")
    ]

# Collect results
results = []

for i, (q, ref) in enumerate(queries):
    try:
        proc = psutil.Process()
        mem_before = proc.memory_full_info().rss / (1024 ** 2)
        t0 = time.perf_counter()

        response = requests.post("http://localhost:8080/get_answer/", json={"query": q})

        t1 = time.perf_counter()
        mem_after = proc.memory_full_info().rss / (1024 ** 2)
        latency_ms = (t1 - t0) * 1000
        mem_diff = mem_after - mem_before

        raw_answer = response.json().get("answer", "")
        cleaned_answer = clean_text(raw_answer)
        cleaned_ref = clean_text(ref)

        tokens = len(cleaned_answer.split())
        throughput = tokens / (t1 - t0) if (t1 - t0) > 0 else 0

        metrics = {
            'EM': None,
            'F1': None,
            'BLEU': None,
            'Semantic_Similarity': None
        }

        if is_valid_text(cleaned_answer) and is_valid_text(cleaned_ref):
            metrics['EM'] = em_metric.compute(predictions=[cleaned_answer], references=[cleaned_ref])['exact_match']
            metrics['F1'] = compute_f1(cleaned_answer, cleaned_ref)
            metrics['BLEU'] = bleu_metric.compute(predictions=[cleaned_answer], references=[[cleaned_ref]])['bleu']
            vec_ref = similarity_model.encode([cleaned_ref])[0]
            vec_pred = similarity_model.encode([cleaned_answer])[0]
            metrics['Semantic_Similarity'] = float(cosine_similarity([vec_pred], [vec_ref])[0][0])

        results.append({
            'query_id': f"Q{i+1}",
            'question': q,
            'answer': raw_answer,
            'latency_ms': latency_ms,
            'mem_MB': mem_diff,
            'throughput_toks_per_s': throughput,
            **metrics
        })

        print(f"✅ Q{i+1} — EM: {metrics['EM']}, F1: {metrics['F1']:.2f}, BLEU: {metrics['BLEU']:.2f}, Sim: {metrics['Semantic_Similarity']:.2f}, Latency: {latency_ms:.2f}ms, Mem: {mem_diff:.3f}MB")

    except Exception as e:
        print(f"❌ Error for 'Q{i+1}': {e}")
        continue

# Convert to DataFrame
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

# Summary and Plotting
if df.empty:
    print("\n===== Summary Statistics =====")
    print("No successful responses were collected. Check your server and input format.")
else:
    print("\n===== Summary Statistics =====")
    print(f"Total Queries: {len(df)}")
    print(f"Avg Latency: {df['latency_ms'].mean():.2f} ms")
    print(f"Avg Memory Usage: {df['mem_MB'].mean():.4f} MB")
    print(f"Avg Throughput: {df['throughput_toks_per_s'].mean():.2f} tokens/sec")
    print(f"Avg EM: {df['EM'].mean():.2f}")
    print(f"Avg F1: {df['F1'].mean():.2f}")
    print(f"Avg BLEU: {df['BLEU'].mean():.2f}")
    print(f"Avg Semantic Similarity: {df['Semantic_Similarity'].mean():.2f}")

    def plot_metric(metric_name, color, ylabel, filename):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['query_id'], df[metric_name], color=color)
        plt.title(f"{ylabel} per Query")
        plt.xlabel("Query")
        plt.ylabel(ylabel)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    output_dir = "Graphs"

    plot_metric('latency_ms', 'skyblue', 'Latency (ms)', f'{output_dir}/graph_latency.png')
    plot_metric('mem_MB', 'mediumseagreen', 'Memory Usage (MB)', f'{output_dir}/graph_memory.png')
    plot_metric('throughput_toks_per_s', 'orchid', 'Token Throughput (tokens/sec)', f'{output_dir}/graph_throughput.png')
    plot_metric('F1', 'gold', 'F1 Score', f'{output_dir}/graph_f1.png')
    plot_metric('BLEU', 'salmon', 'BLEU Score', f'{output_dir}/graph_bleu.png')
    plot_metric('Semantic_Similarity', 'plum', 'Semantic Similarity', f'{output_dir}/graph_similarity.png')
