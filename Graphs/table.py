import matplotlib.pyplot as plt
import pandas as pd

# Define the queries
queries = [
    "What are some good postpartum exercises?",
    "What foods should I avoid while breastfeeding?",
    "When can I start exercising after giving birth?",
    "How often should I feed my newborn?",
    "What are common signs of postpartum depression?",
    "How do I know if my baby is getting enough milk?",
    "What is the best way to soothe a colicky baby?",
    "When should I schedule my baby's first doctor appointment?",
    "Is it normal to feel overwhelmed after childbirth?",
    "What are baby blues and how are they different from postpartum depression?",
    "How can I increase my breast milk supply?",
    "How much sleep does a newborn need?",
    "Can I take painkillers while breastfeeding?",
    "What are signs of a good latch during breastfeeding?"
]

# Create a DataFrame with QIDs
df = pd.DataFrame({
    "Short ID": [f"Q{i+1}" for i in range(len(queries))],
    "Full Question": queries
})

# Plot the table and save as PNG
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.4, 1.4)
plt.tight_layout()
plt.savefig("qid_table.png", dpi=300)
plt.show()
