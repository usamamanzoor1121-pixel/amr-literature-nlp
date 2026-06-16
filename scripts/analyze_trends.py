"""
Script 3: Trend analysis and insight extraction — complete 41-topic version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

FIGURE_DIR = "figures"
DATA_DIR   = "data/processed"
os.makedirs(FIGURE_DIR, exist_ok=True)

print("Loading data...")
topic_info  = pd.read_csv(f"{DATA_DIR}/topic_info.csv")
doc_topics  = pd.read_csv(f"{DATA_DIR}/document_topics.csv")

doc_topics["year"] = pd.to_numeric(doc_topics["year"], errors="coerce")
doc_topics = doc_topics[doc_topics["year"].between(2000, 2024)].copy()

TOPIC_LABELS = {
    0:  "Klebsiella & Carbapenem Resistance",
    1:  "E. coli & Colistin (mcr genes)",
    2:  "Environmental AMR (Wastewater)",
    3:  "Hospital Surveillance Networks",
    4:  "ESKAPE Mechanisms & Therapeutics",
    5:  "XDR Typhoid (Pakistan/South Asia)",
    6:  "Acinetobacter baumannii (OXA genes)",
    7:  "Antibiotic Knowledge & Stewardship",
    8:  "Beta-lactamase Inhibitors",
    9:  "Pseudomonas aeruginosa",
    10: "AMR Surveillance (India/South Asia)",
    11: "Urinary Tract Infections",
    12: "Bloodstream Infections (ESKAPE)",
    13: "ML/AI for AMR Prediction",
    14: "Nanoparticles & Novel Agents",
    15: "ESBL & Carbapenemase Phenotypes",
    16: "MRSA & Livestock/Mastitis",
    17: "Biofilm & Quorum Sensing",
    18: "COVID-19 & AMR Co-infections",
    19: "Antimicrobial Peptides (AMPs)",
    20: "Phage Therapy",
    21: "Carbapenemase & OXA-48",
    22: "Antibiotic Prescribing Practices",
    23: "ETEC & Diarrheal Disease",
    24: "Aquaculture & Vibrio AMR",
    25: "Salmonella & Poultry AMR",
    26: "Ventilator-Associated Pneumonia",
    27: "MRSA Clinical Isolates",
    28: "Shigella Resistance",
    29: "Wound & Surgical Infections",
    30: "Plant Extracts Antibacterial",
    31: "NDM Carbapenemase",
    32: "Cholera & Vibrio cholerae",
    33: "Enterococcus faecium/faecalis",
    34: "CPE & CRE Surveillance",
    35: "UPEC & ESBL UTI",
    36: "Neonatal Sepsis & AMR",
    37: "Agricultural AMU & Farmers",
    38: "Carbapenem-Resistant Gram-Negatives",
    39: "ESBL Carriage & Enterobacteriaceae",
    40: "Enterobacter & NDM",
}

# ── Topic growth ───────────────────────────────────────────────────────────────

recent = doc_topics[doc_topics["year"] >= 2015].copy()
early  = doc_topics[doc_topics["year"].between(2005, 2014)].copy()

recent_counts = recent[recent["topic"] >= 0]["topic"].value_counts(normalize=True) * 100
early_counts  = early[early["topic"] >= 0]["topic"].value_counts(normalize=True) * 100

growth = pd.DataFrame({
    "early_pct":  early_counts,
    "recent_pct": recent_counts,
}).fillna(0)
growth["growth"] = growth["recent_pct"] - growth["early_pct"]
growth["label"]  = growth.index.map(TOPIC_LABELS).fillna("Unknown")
growth = growth.sort_values("growth", ascending=False)

# Figure 1: Topic growth
fig, ax = plt.subplots(figsize=(14, 12))
colors = ["#e63946" if x > 0 else "#2a9d8f" for x in growth["growth"]]
ax.barh(range(len(growth)), growth["growth"], color=colors)
ax.set_yticks(range(len(growth)))
ax.set_yticklabels(growth["label"], fontsize=8)
ax.set_xlabel("Change in topic share (%, 2015-2024 vs 2005-2014)")
ax.set_title("Emerging vs Declining AMR Research Topics (2015-2024 vs 2005-2014)")
ax.axvline(0, color="black", linewidth=0.8)
ax.invert_yaxis()
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#e63946", label="Growing"),
    Patch(facecolor="#2a9d8f", label="Declining"),
], loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/topic_growth.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_DIR}/topic_growth.png")

# Figure 2: South Asia vs Global
sa = doc_topics[doc_topics["query"]=="amr_south_asia"]
gl = doc_topics[doc_topics["query"]=="amr_eskape_global"]

sa_pct = sa[sa["topic"]>=0]["topic"].value_counts(normalize=True) * 100
gl_pct = gl[gl["topic"]>=0]["topic"].value_counts(normalize=True) * 100

compare = pd.DataFrame({
    "South Asia": sa_pct,
    "Global ESKAPE": gl_pct,
}).fillna(0)
compare.index = compare.index.map(TOPIC_LABELS).fillna("Unknown")
compare["SA_excess"] = compare["South Asia"] - compare["Global ESKAPE"]
compare = compare.sort_values("SA_excess", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(compare))
width = 0.35
ax.barh(x + width/2, compare["South Asia"], width, label="South Asia", color="#e76f51")
ax.barh(x - width/2, compare["Global ESKAPE"], width, label="Global ESKAPE", color="#457b9d")
ax.set_yticks(x)
ax.set_yticklabels(compare.index, fontsize=9)
ax.set_xlabel("Topic share (% of documents)")
ax.set_title("South Asia vs Global: Top 15 Topic Differences")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/south_asia_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_DIR}/south_asia_profile.png")

# Figure 3: Publication trend
yearly = doc_topics.groupby("year").size().reset_index(name="count")
yearly = yearly[yearly["year"] >= 2000]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(yearly["year"], yearly["count"], color="#2a9d8f", edgecolor="white", width=0.7)
z = np.polyfit(yearly["year"], yearly["count"], 2)
p = np.poly1d(z)
x_line = np.linspace(2000, 2024, 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend")
ax.set_xlabel("Year")
ax.set_ylabel("Publications")
ax.set_title("AMR Literature Growth 2000-2024")
ax.set_xticks(range(2000, 2025, 2))
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/publication_growth.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_DIR}/publication_growth.png")

# ── Key findings ───────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print(f"\nTotal abstracts: {len(doc_topics):,}")
print(f"Topics found: {len(TOPIC_LABELS)}")
print(f"Year range: {int(doc_topics['year'].min())} - {int(doc_topics['year'].max())}")

print("\nTop 5 GROWING topics:")
for _, row in growth.head(5).iterrows():
    print(f"  {row['label']}: +{row['growth']:.2f}%")

print("\nTop 5 DECLINING topics:")
for _, row in growth.tail(5).iterrows():
    print(f"  {row['label']}: {row['growth']:.2f}%")

sa_specific = compare[compare["SA_excess"] > 1.0]
print("\nSouth Asia overrepresented topics:")
for idx, row in sa_specific.iterrows():
    print(f"  {idx}: SA={row['South Asia']:.1f}%, Global={row['Global ESKAPE']:.1f}%")
