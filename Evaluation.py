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
    ("What are common signs of postpartum depression?", "Persistent sadness, low energy, anxiety, changes in sleep or appetite, feelings of guilt or worthlessness, and difficulty bonding with the baby."),
    ("How do I know if my baby is getting enough milk?", "Signs include regular weight gain, feeding 8 to 12 times in 24 hours, hearing swallowing sounds during feeds, and having at least six wet diapers and three bowel movements daily by the fifth day. Your baby should also appear content after feeding. [Source: CDC]"),
    ("What is the best way to soothe a colicky baby?", "Effective methods include holding or cuddling your baby, gentle rocking, using a pacifier, swaddling, giving a warm bath, and providing white noise. It's also helpful to ensure your baby is not hungry and to burp them after feeds. [Source: Mayo Clinic]"),
    ("When should I schedule my baby's first doctor appointment?", "The first pediatrician visit should occur within 3 to 5 days after birth. This checkup assesses your baby's weight, feeding, and overall health. [Source: HealthPartners]"),
    ("Is it normal to feel overwhelmed after childbirth?", "Yes, experiencing a range of emotions, including feeling overwhelmed, is common due to hormonal changes, sleep deprivation, and the responsibilities of caring for a newborn. If these feelings persist or intensify, consult your healthcare provider. [Source: NHS]"),
    ("What are baby blues and how are they different from postpartum depression?", "Baby blues involve mood swings, crying spells, and anxiety, typically resolving within two weeks postpartum. Postpartum depression is more severe, with symptoms lasting longer and potentially interfering with daily life. If symptoms persist beyond two weeks, seek medical advice. [Source: NHS]"),
    ("How can I increase my breast milk supply?", "To increase breast milk supply, ensure frequent breastfeeding or pumping, maintain a healthy diet, stay hydrated, and get adequate rest. Consulting a lactation consultant can also provide personalized strategies. [Source: La Leche League International]"),
    ("How much sleep does a newborn need?", "Newborns typically sleep about 16 to 17 hours per day, divided into short periods of 2 to 4 hours. Their sleep patterns are irregular in the first few weeks. [Source: American Academy of Pediatrics]"),
    ("Can I take painkillers while breastfeeding?", "Some painkillers, like acetaminophen and ibuprofen, are generally considered safe during breastfeeding. However, always consult your healthcare provider before taking any medication. [Source: NHS]"),
    ("What are signs of a good latch during breastfeeding?", "Signs include baby's mouth covering a large portion of the areola, rhythmic sucking and swallowing, and absence of pain for the mother. The baby's lips should be flanged outward. [Source: La Leche League International]"),
    ("When does a baby start smiling socially?", "Babies typically begin to smile socially between 6 to 8 weeks of age, responding to familiar faces and voices. [Source: CDC]"),
    ("How can I care for my C-section incision at home?", "Keep the incision clean and dry, avoid strenuous activities, and watch for signs of infection like redness or discharge. Follow your healthcare provider's instructions for wound care. [Source: NHS]"),
    ("Is it okay for my baby to sleep on their side?", "No, babies should always be placed on their backs to sleep to reduce the risk of sudden infant death syndrome (SIDS). Side sleeping is not recommended. [Source: NHS]"),
    ("When can I start pumping breast milk?", "You can start pumping breast milk as soon as you feel comfortable. Some mothers begin within the first few days, especially if separated from their baby or to relieve engorgement. [Source: La Leche League International]"),
    ("What should I pack in my postpartum hospital bag?", "Essentials include comfortable clothing, toiletries, maternity pads, nursing bras, snacks, phone charger, and items for the baby like clothes and diapers. [Source: NHS]"),
    ("How do I deal with postpartum hair loss?", "Postpartum hair loss is common due to hormonal changes. It usually resolves within a few months. Maintaining a healthy diet and gentle hair care can help. [Source: American Academy of Dermatology]"),
    ("How soon after birth can I get pregnant again?", "It's possible to become pregnant as soon as three weeks after childbirth, even if menstruation hasn't resumed. Discuss contraception options with your healthcare provider. [Source: NHS]"),
    ("What are some good postpartum exercises?", "Gentle exercises like walking, pelvic floor exercises, and stretching are beneficial postpartum. Always consult your healthcare provider before starting any exercise routine. [Source: NHS]"),
    ("Can I drink coffee while breastfeeding?", "Moderate caffeine intake (about 200-300 mg per day) is generally considered safe while breastfeeding. Monitor your baby for any signs of sensitivity. [Source: NHS]"),
    ("What are signs that my newborn might be sick?", "Signs include fever, lethargy, poor feeding, difficulty breathing, or unusual rashes. If you notice any of these, contact your healthcare provider immediately. [Source: NHS]"),
    ("What vaccinations does my baby need in the first year?", "Vaccinations typically include those for hepatitis B, rotavirus, DTaP, Hib, pneumococcal, polio, influenza, MMR, and varicella, following the recommended schedule. [Source: CDC]"),
    ("How do I manage nighttime feedings more easily?", "Establish a calming bedtime routine, keep the lights dim during feedings, and have necessary supplies within reach to make nighttime feedings smoother. [Source: NHS]"),
    ("Is it normal to have mood swings after childbirth?", "Yes, mood swings are common due to hormonal changes. If mood swings persist or worsen, consult your healthcare provider. [Source: NHS]"),
    ("How can I support my mental health postpartum?", "Maintain open communication with loved ones, get adequate rest, eat healthily, and seek support from healthcare providers or support groups as needed. [Source: NHS]"),
    ("Can I use tampons during postpartum bleeding?", "It's recommended to avoid tampons during postpartum bleeding to reduce the risk of infection. Use sanitary pads instead. [Source: NHS]"),
    ("When should I call a doctor after giving birth?", "Contact your healthcare provider if you experience heavy bleeding, severe pain, signs of infection, or any concerning symptoms. [Source: NHS]"),
    ("How do I relieve sore breasts during postpartum?", "To alleviate sore breasts, apply warm compresses, ensure proper latching during breastfeeding, and wear a supportive bra. Over-the-counter pain relievers may also help. [Source: UnityPoint Health]"),
    ("What is kangaroo care and its benefits?", "Kangaroo care involves skin-to-skin contact between a parent and newborn, promoting bonding, regulating the baby's temperature, and supporting breastfeeding. [Source: Wikipedia]"),
    ("How often should I bathe my newborn?", "Newborns don't need daily baths; bathing them 2-3 times a week is sufficient. Sponge baths are recommended until the umbilical cord falls off. [Source: Integris Health]"),
    ("Can I overfeed my newborn?", "While rare, overfeeding can occur, especially with bottle-fed babies. Signs include spitting up, gas, and irritability. Feeding on demand and recognizing hunger cues can help prevent overfeeding. [Source: Integris Health]"),
    ("What are common mistakes new parents make with newborns?", "Common mistakes include overbundling the baby, not following safe sleep practices, and comparing their baby to others. Trusting instincts and seeking guidance when needed is essential. [Source: The Sun]"),
    ("How do I care for my baby's umbilical cord stump?", "Keep the stump clean and dry, and allow it to fall off naturally, usually within 1-2 weeks. Avoid submerging it in water until it detaches. [Source: Parents.com]"),
    ("Is it normal for my newborn to have hiccups?", "Yes, hiccups are common in newborns and usually harmless. They often resolve on their own without intervention. [Source: Norton Children's]"),
    ("How can I tell if my baby is too hot or cold?", "Feel the baby's neck or back to assess temperature. Signs of overheating include sweating and flushed cheeks, while cold babies may have cool extremities. [Source: NHS]"),
    ("When can I start tummy time with my baby?", "Tummy time can begin as early as the first day home from the hospital. Start with short sessions, gradually increasing as the baby grows stronger. [Source: NHS]"),
    ("What is the best way to burp my baby?", "Hold the baby upright against your chest or seated on your lap, supporting the head and neck, and gently pat or rub the back to release air. [Source: Integris Health]"),
    ("How do I know if my baby has jaundice?", "Jaundice causes a yellowing of the skin and eyes. It's common in newborns and usually harmless, but persistent or severe cases require medical attention. [Source: NHS]"),
    ("What should I do if my baby has a fever?", "For newborns under 3 months, a fever of 100.4°F (38°C) or higher requires immediate medical attention. Monitor for other symptoms and consult a healthcare provider. [Source: NHS]"),
    ("How can I soothe my baby's diaper rash?", "Keep the area clean and dry, change diapers frequently, and apply a barrier cream or ointment. Allowing diaper-free time can also help. [Source: NHS]"),
    ("When should I introduce a pacifier to my baby?", "It's recommended to wait until breastfeeding is well-established, usually around 3-4 weeks, before introducing a pacifier. [Source: NHS]"),
    ("How do I handle cluster feeding?", "Cluster feeding is normal, especially during growth spurts. Feed on demand, ensure you're comfortable, and stay hydrated and nourished. [Source: NHS]"),
    ("What are signs of a growth spurt in my baby?", "Increased feeding frequency, fussiness, and disrupted sleep patterns can indicate a growth spurt, typically occurring at 2-3 weeks, 6 weeks, and 3 months. [Source: NHS]"),
    ("How can I prevent SIDS (Sudden Infant Death Syndrome)?", "Place your baby on their back to sleep, use a firm mattress, keep the sleep area free of soft items, and avoid overheating. [Source: NHS]"),
    ("Is it safe to co-sleep with my baby?", "Co-sleeping increases the risk of SIDS. It's safer to have the baby sleep in a separate crib or bassinet in the same room. [Source: NHS]"),
    ("When will my baby start teething?", "Teething typically begins around 6 months, but it can vary. Signs include drooling, gum swelling, and increased irritability. [Source: NHS]"),
    ("How do I care for my baby's nails?", "Use baby nail clippers or a file to trim nails regularly, preventing scratches. It's best to do this when the baby is calm or asleep. [Source: NHS]"),
    ("What are signs of dehydration in my baby?", "Signs include fewer wet diapers, dry mouth, sunken eyes, and lethargy. Seek medical attention if you suspect dehydration. [Source: NHS]"),
    ("How can I tell if my baby is constipated?", "Infrequent bowel movements, hard stools, and straining can indicate constipation. Consult a healthcare provider for guidance. [Source: NHS]"),
    ("When should I start reading to my baby?", "You can start reading to your baby from birth. It promotes bonding and language development. [Source: NHS]"),
    ("How do I handle my baby's spit-up?", "Spitting up is common and usually harmless. Keep the baby upright after feeding and ensure proper burping. [Source: NHS]"),
    ("What is tummy time and why is it important?", "Tummy time involves placing your baby on their stomach while awake to strengthen neck and shoulder muscles, preventing flat spots on the head. [Source: NHS]"),
    ("How can I tell if my baby has colic?", "Colic is characterized by prolonged crying episodes, often in the evening, without an apparent cause. Consult a healthcare provider for management strategies. [Source: NHS]"),
    ("When will my baby start crawling?", "Babies typically start crawling between 6 to 10 months, but some may skip crawling and move directly to standing or walking. [Source: NHS]"),
    ("How do I transition my baby to solid foods?", "Introduce solid foods around 6 months, starting with pureed fruits and vegetables, while continuing breastfeeding or formula. [Source: NHS]"),
    ("What vaccinations does my baby need?", "Vaccination schedules vary by country, but common vaccines include those for hepatitis B, DTaP, polio, and MMR. Consult your healthcare provider for the recommended schedule. [Source: NHS]")
]

# -----------------------------
# Helper functions
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
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

# -----------------------------
# Summary stats
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
