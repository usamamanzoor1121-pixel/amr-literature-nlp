"""
Script 2: BERTopic modeling on AMR abstracts
- Preprocess abstracts
- Fit BERTopic with sentence-transformers
- Extract topics, trends over time, per-query differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import warnings
warnings.filterwarnings("ignore")

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

OUTPUT_DIR = "data/processed"
FIGURE_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading abstracts...")
df = pd.read_csv("data/raw/amr_abstracts.csv")
df = df[df["abstract"].notna() & (df["abstract"].str.len() > 100)].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].between(2000, 2024)].copy()
df = df.reset_index(drop=True)
print(f"Abstracts for modeling: {len(df)}")

# ── Combine title + abstract for richer representation ─────────────────────────

df["text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")

# ── BERTopic setup ─────────────────────────────────────────────────────────────

print("\nInitializing BERTopic components...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_features=10000,
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    verbose=True,
)

# ── Fit model ──────────────────────────────────────────────────────────────────

print("\nFitting BERTopic (this takes 5-10 minutes)...")
texts = df["text"].tolist()
topics, probs = topic_model.fit_transform(texts)
df["topic"] = topics

# ── Topic summary ──────────────────────────────────────────────────────────────

topic_info = topic_model.get_topic_info()
print(f"\nTopics found: {len(topic_info)-1} (excluding outliers)")
print("\nTop 20 topics by size:")
print(topic_info.head(21).to_string(index=False))

# Save topic info
topic_info.to_csv(f"{OUTPUT_DIR}/topic_info.csv", index=False)

# Save topic keywords
topic_keywords = {}
for topic_id in topic_info["Topic"].tolist():
    if topic_id == -1:
        continue
    words = topic_model.get_topic(topic_id)
    if words:
        topic_keywords[topic_id] = [w[0] for w in words[:10]]

keywords_df = pd.DataFrame([
    {"topic_id": k, "keywords": ", ".join(v)}
    for k, v in topic_keywords.items()
])
keywords_df.to_csv(f"{OUTPUT_DIR}/topic_keywords.csv", index=False)

# Save document-topic assignments
df[["pmid","title","year","query","topic"]].to_csv(
    f"{OUTPUT_DIR}/document_topics.csv", index=False
)

# ── Topic trends over time ──────────────────────────────────────────────────────

print("\nComputing topic trends over time...")
timestamps = df["year"].tolist()

try:
    topics_over_time = topic_model.topics_over_time(
        texts, timestamps,
        global_tuning=True,
        evolution_tuning=True,
        nr_bins=10,
    )
    topics_over_time.to_csv(f"{OUTPUT_DIR}/topics_over_time.csv", index=False)
    print(f"Saved topics over time: {len(topics_over_time)} rows")
except Exception as e:
    print(f"Topics over time failed: {e}")

# ── Figure 1: Topic size bar chart ─────────────────────────────────────────────

top_topics = topic_info[topic_info["Topic"] != -1].head(20)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(
    range(len(top_topics)),
    top_topics["Count"],
    color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_topics)))
)
ax.set_yticks(range(len(top_topics)))
ax.set_yticklabels([
    f"Topic {row['Topic']}: {row['Name'][:50]}"
    for _, row in top_topics.iterrows()
], fontsize=8)
ax.set_xlabel("Number of Documents")
ax.set_title("Top 20 BERTopic Topics in AMR Literature (2000-2024)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/topic_sizes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_DIR}/topic_sizes.png")

# ── Figure 2: South Asia vs Global topic distribution ──────────────────────────

sa_docs = df[df["query"]=="amr_south_asia"]["topic"].value_counts().head(10)
global_docs = df[df["query"]=="amr_eskape_global"]["topic"].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, data, title in zip(axes,
    [sa_docs, global_docs],
    ["South Asia AMR Literature", "Global ESKAPE AMR Literature"]):
    
    topic_labels = [
        f"T{t}: {topic_model.get_topic(t)[0][0] if topic_model.get_topic(t) else '?'}"
        for t in data.index if t != -1
    ]
    counts = [data[t] for t in data.index if t != -1]
    
    ax.barh(range(len(counts)), counts,
             color=plt.cm.plasma(np.linspace(0.2, 0.8, len(counts))))
    ax.set_yticks(range(len(topic_labels)))
    ax.set_yticklabels(topic_labels, fontsize=9)
    ax.set_xlabel("Documents")
    ax.set_title(title)
    ax.invert_yaxis()

plt.suptitle("Topic Distribution: South Asia vs Global AMR Literature", fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/south_asia_vs_global.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_DIR}/south_asia_vs_global.png")

# ── Figure 3: Publication trend by year ────────────────────────────────────────

yearly = df.groupby("year").size().reset_index(name="count")
yearly = yearly[yearly["year"] >= 2000]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(yearly["year"], yearly["count"], color="#2a9d8f", edgecolor="white")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Publications")
ax.set_title("AMR Literature Growth 2000-2024 (ESKAPE + South Asia)")
ax.set_xticks(range(2000, 2025, 2))
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/publication_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_DIR}/publication_trend.png")

print("\nDone. All outputs saved.")
