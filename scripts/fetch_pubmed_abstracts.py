"""
Script 1: Fetch AMR PubMed abstracts for ESKAPE pathogens
Focus: South Asia + global AMR literature 2000-2024
"""

from Bio import Entrez
import pandas as pd
import time
import os
import json

Entrez.email = "usama.manzoor1121@gmail.com"

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Search queries — broad AMR + ESKAPE specific
QUERIES = {
    "amr_eskape_global": (
        '("antimicrobial resistance" OR "antibiotic resistance" OR "AMR") '
        'AND ("Klebsiella pneumoniae" OR "Acinetobacter baumannii" OR '
        '"Staphylococcus aureus" OR "Pseudomonas aeruginosa" OR '
        '"Enterococcus faecium" OR "Enterobacter" OR "ESKAPE") '
        'AND ("2000"[PDAT]:"2024"[PDAT]) '
        'AND hasabstract'
    ),
    "amr_south_asia": (
        '("antimicrobial resistance" OR "antibiotic resistance") '
        'AND ("Pakistan" OR "India" OR "Bangladesh" OR "South Asia") '
        'AND ("2000"[PDAT]:"2024"[PDAT]) '
        'AND hasabstract'
    ),
    "amr_mechanisms": (
        '("ESBL" OR "carbapenem resistance" OR "mcr" OR "NDM" OR "KPC" OR "OXA-48") '
        'AND ("antimicrobial resistance" OR "antibiotic resistance") '
        'AND ("2000"[PDAT]:"2024"[PDAT]) '
        'AND hasabstract'
    ),
}

MAX_PER_QUERY = 2000

def fetch_abstracts(query_name, query, max_results=2000):
    print(f"\nSearching: {query_name}")
    
    # Search
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    print(f"  Found {record['Count']} total, fetching {len(ids)}")

    if not ids:
        return pd.DataFrame()

    # Fetch in batches
    records = []
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        print(f"  Fetching batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}...")
        try:
            handle = Entrez.efetch(
                db="pubmed", id=batch,
                rettype="xml", retmode="xml"
            )
            from Bio import Medline
            handle2 = Entrez.efetch(
                db="pubmed", id=batch,
                rettype="medline", retmode="text"
            )
            for rec in Medline.parse(handle2):
                abstract = rec.get("AB", "")
                title = rec.get("TI", "")
                if not abstract:
                    continue
                records.append({
                    "pmid": rec.get("PMID", ""),
                    "title": title,
                    "abstract": abstract,
                    "year": rec.get("DP", "")[:4] if rec.get("DP") else "",
                    "journal": rec.get("JT", ""),
                    "authors": "; ".join(rec.get("AU", [])[:3]),
                    "mesh_terms": "; ".join(rec.get("MH", [])[:10]),
                    "query": query_name,
                })
            handle2.close()
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(0.5)

    return pd.DataFrame(records)

all_dfs = []
for query_name, query in QUERIES.items():
    df = fetch_abstracts(query_name, query, MAX_PER_QUERY)
    all_dfs.append(df)
    print(f"  Got {len(df)} abstracts with text")
    time.sleep(2)

combined = pd.concat(all_dfs, ignore_index=True)

# Deduplicate by PMID
before = len(combined)
combined = combined.drop_duplicates(subset=["pmid"])
print(f"\nDeduplication: {before} -> {len(combined)} unique abstracts")

# Filter out empty abstracts
combined = combined[combined["abstract"].str.len() > 100]
print(f"After length filter: {len(combined)} abstracts")

# Year distribution
print("\nYear distribution (top 10):")
print(combined["year"].value_counts().head(10).to_string())

print("\nQuery distribution:")
print(combined["query"].value_counts().to_string())

out_path = os.path.join(OUTPUT_DIR, "amr_abstracts.csv")
combined.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
