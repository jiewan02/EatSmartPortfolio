"""
Run this script once from inside the model-server/ folder to generate
recipe_embeddings.npy from dataset_preprocessed.csv.

Usage:
    cd model-server
    python ../data-pipeline/generate_embeddings.py
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"

CSV_PATH = "dataset_preprocessed.csv"
OUTPUT_PATH = "recipe_embeddings.npy"

df = pd.read_csv(CSV_PATH)
texts = df["요리명"].fillna("").tolist()

print(f"Generating embeddings for {len(texts)} recipes...")

embeddings = []
for i, text in enumerate(texts):
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        embeddings.append(resp.data[0].embedding)
    except Exception as e:
        print(f"  [ERROR] row {i}: {e} — using zero vector")
        embeddings.append([0.0] * 1536)

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(texts)} done")
        time.sleep(0.2)

arr = np.array(embeddings, dtype="float32")
np.save(OUTPUT_PATH, arr)
print(f"\nSaved {arr.shape[0]} embeddings to {OUTPUT_PATH}")
