from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import torch
import requests
from bs4 import BeautifulSoup
import json
from graph_similarity_v2 import RecipeGraphSimilarity
from jiewan_model_v2 import graph_rag_search_with_scoring_explanation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
client = OpenAI()

app = Flask(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

similarity_service = RecipeGraphSimilarity(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

embeddings = np.load("recipe_embeddings.npy").astype("float32")
emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_norm_t = torch.from_numpy(emb_norm).to(DEVICE)

df = pd.read_csv("dataset_preprocessed.csv")

EMBED_MODEL = "text-embedding-3-small"


def embed_query(text: str, keywords: dict) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / np.linalg.norm(v)
    return torch.from_numpy(v).to(DEVICE)


def get_recipe(id):
    res = requests.get(f"https://www.10000recipe.com/recipe/{id}")
    soup = BeautifulSoup(res.content, features="html.parser")
    main_title = soup.select_one("div.view2_summary h3").text

    infos = soup.select_one("div.view2_summary_info")
    info1 = infos.select_one("span.view2_summary_info1")
    info1 = info1.text if info1 else ""
    info2 = infos.select_one("span.view2_summary_info2")
    info2 = info2.text if info2 else ""
    info3 = infos.select_one("span.view2_summary_info3")
    info3 = info3.text if info3 else ""

    image_url = ""
    img_tag = soup.select_one("div.centeredcrop img")
    if not img_tag:
        img_tag = soup.select_one("div.view3_pic img")
    if img_tag and img_tag.get("src"):
        image_url = img_tag["src"].strip()

    grid = soup.select_one("div.cont_ingre2")
    result = {"재료": [], "조리도구": []}

    big_sections = grid.select("div.best_tit")
    for section in big_sections:
        title = section.get_text(strip=True)
        next_div = section.find_next_sibling("div", class_="ready_ingre3")
        if not next_div:
            continue
        if "재료" in title:
            for li in next_div.select("li"):
                name_tag = li.select_one("div.ingre_list_name a")
                if not name_tag:
                    continue
                name = name_tag.get_text(strip=True)
                qty_tag = li.select_one("span.ingre_list_ea")
                qty = qty_tag.get_text(strip=True) if qty_tag else ""
                result["재료"].append((name, qty))
        elif "조리도구" in title:
            for li in next_div.select("li"):
                name_tag = li.select_one("div.ingre_list_name")
                if name_tag:
                    result["조리도구"].append(name_tag.get_text(strip=True))

    steps = []
    for cont in soup.select("div.view_step_cont"):
        step = {"text": "", "tools": "", "img_url": ""}
        body = cont.select_one("div.media-body")
        if body:
            main_text_node = body.find(string=True, recursive=False)
            if main_text_node:
                step["text"] = main_text_node.strip()
                tools = body.select("p")
                if tools:
                    step["tools"] = tools[0].text.strip()
        img = cont.find("div", id=re.compile(r"stepimg\d+")).select_one("img")
        if img:
            step["img_url"] = img["src"]
        steps.append(step)

    return {"title": main_title, "infos": [info1, info2, info3], "image_url": image_url, "steps": steps, "grid_info": result}


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    keywords = (data.get("matchedKeywords") or {})
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400

    q = embed_query(query, keywords)
    sims_t = emb_norm_t @ q
    k = min(top_k, sims_t.shape[0])
    scores_t, idxs_t = torch.topk(sims_t, k)
    idxs = idxs_t.cpu().numpy()
    scores = scores_t.cpu().numpy().tolist()

    results = []
    for idx, score in zip(idxs, scores):
        row = df.iloc[idx]
        results.append({
            "index": int(idx),
            "score": float(score),
            "name": row.get("요리명", ""),
            "types": row.get("요리종류별명", []),
            "intro": row.get("요리소개_cleaned", ""),
            "servings": row.get("요리인분명", ""),
            "difficulty": row.get("요리난이도명", ""),
            "time": row.get("요리시간명", ""),
            "ingredients": row.get("재료", []),
        })

    return jsonify({"results": results})


@app.route("/jiewan-search-v2", methods=["POST"])
def graph_search_endpoint():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    filterKeywords = (data.get("filterKeywords") or {})
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        res = graph_rag_search_with_scoring_explanation(query, filterKeywords=filterKeywords, top_k=top_k)
    except Exception as e:
        print("[ERROR] graph_rag_search failed:", e)
        return jsonify({"error": "graph search failed", "detail": str(e)}), 500

    return jsonify({
        "results": res["recipes"],
        "keywords": res["keywords"],
    })


@app.route("/crawl-recipe/<int:recipe_id>", methods=["GET"])
def crawl_recipe_endpoint(recipe_id):
    try:
        result = similarity_service.get_similar_recipes(recipe_id=recipe_id, top_n=3, min_shared_ings=2)
        data = get_recipe(recipe_id)
        return jsonify({
            "id": recipe_id,
            "data": data,
            "overall": result["overall"],
            "ingredients": result["ingredients"],
        })
    except Exception as e:
        print("[ERROR] get_recipe failed:", e)
        return jsonify({"error": "crawl_failed", "detail": str(e)}), 500


@app.route("/similar-recipes", methods=["POST"])
def similar_recipes_endpoint():
    data = request.get_json() or {}
    recipe_id = data.get("recipe_id")

    try:
        result = similarity_service.get_similar_recipes(recipe_id=recipe_id, top_n=3, min_shared_ings=2)
    except Exception as e:
        print("[ERROR] similar_recipes failed:", e)
        return jsonify({"error": "similar_recipes failed", "detail": str(e)}), 500

    return jsonify({
        "recipe_id": recipe_id,
        "overall": result["overall"],
        "ingredients": result["ingredients"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
