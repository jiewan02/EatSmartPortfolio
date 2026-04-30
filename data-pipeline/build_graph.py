# build_graph.py
from neo4j import GraphDatabase
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
if not PASSWORD:
    raise ValueError("NEO4J_PASSWORD is not set. Add it to a .env file in this folder.")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def parse_servings(serv_str: str):
    if not isinstance(serv_str, str):
        return None
    num = "".join(ch for ch in serv_str if ch.isdigit())
    return int(num) if num else None

def parse_time_to_min(time_str: str):
    if not isinstance(time_str, str):
        return None
    s = time_str.strip()
    if "분" in s:
        num = "".join(ch for ch in s if ch.isdigit())
        return int(num) if num else None
    if "시간" in s:
        num = "".join(ch for ch in s if ch.isdigit())
        return int(num) * 60 if num else None
    return None

def safe_list_parse(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []
    s = x.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",")]
    return [s]

def create_recipe_tx(tx, recipe_id, title, name, views,
                     intro, servings, difficulty, time_min,
                     full_text, image_url,
                     method, situation, categories,
                     main_ingredients, ingredients):

    # 1) Recipe 노드
    tx.run("""
    MERGE (r:Recipe {recipe_id: $recipe_id})
    SET r.title = $title,
        r.name = $name,
        r.views = $views,
        r.intro = $intro,
        r.servings = $servings,
        r.difficulty = $difficulty,
        r.time_min = $time_min,
        r.full_text = $full_text,
        r.image_url = $image_url
    """, recipe_id=recipe_id, title=title, name=name, views=views,
         intro=intro, servings=servings, difficulty=difficulty,
         time_min=time_min, full_text=full_text, image_url=image_url)

    # 2) Method
    if method:
        tx.run("""
        MERGE (m:Method {name: $method})
        WITH m
        MATCH (r:Recipe {recipe_id: $recipe_id})
        MERGE (r)-[:COOKED_BY]->(m)
        """, method=method, recipe_id=recipe_id)

    # 3) Situation
    if situation:
        tx.run("""
        MERGE (s:Situation {name: $situation})
        WITH s
        MATCH (r:Recipe {recipe_id: $recipe_id})
        MERGE (r)-[:FOR_SITUATION]->(s)
        """, situation=situation, recipe_id=recipe_id)

    # 4) Category
    for cat in categories or []:
        tx.run("""
        MERGE (c:Category {name: $cat})
        WITH c
        MATCH (r:Recipe {recipe_id: $recipe_id})
        MERGE (r)-[:IN_CATEGORY]->(c)
        """, cat=cat, recipe_id=recipe_id)

    # 5) Main ingredients
    for mi in main_ingredients or []:
        tx.run("""
        MERGE (i:Ingredient {name: $ing})
        WITH i
        MATCH (r:Recipe {recipe_id: $recipe_id})
        MERGE (r)-[:HAS_INGREDIENT]->(i)
        """, ing=mi, recipe_id=recipe_id)

    # 6) 전체 재료
    for ing in ingredients or []:
        tx.run("""
        MERGE (i:Ingredient {name: $ing})
        WITH i
        MATCH (r:Recipe {recipe_id: $recipe_id})
        MERGE (r)-[:HAS_INGREDIENT]->(i)
        """, ing=ing, recipe_id=recipe_id)

def build_graph_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)  # 인코딩, sep 필요하면 수정

    with driver.session() as session:
        for _, row in df.iterrows():
            recipe_id = int(row["레시피일련번호"])
            title = str(row["레시피제목"])
            name = str(row["요리명"])
            views = int(row.get("조회수", 0))

            method = str(row.get("요리방법설명", "")).strip() or None
            situation = str(row.get("요리상황설명", "")).strip() or None

            categories = safe_list_parse(row.get("요리종류별명", "[]"))
            main_ingredients = safe_list_parse(row.get("요리재료별명", "[]"))
            ingredients = safe_list_parse(row.get("재료", "[]"))

            intro = str(row.get("요리소개", "")).strip()
            ingred_text = str(row.get("요리재료내용", "")).strip()
            image_url = str(row.get("이미지링크", "")).strip()

            servings = parse_servings(row.get("요리인분명", None))
            difficulty = str(row.get("요리난이도명", "")).strip() or None
            time_min = parse_time_to_min(row.get("요리시간명", None))

            full_text = f"{title}\n{name}\n{intro}\n{ingred_text}"

            session.execute_write(
                create_recipe_tx,
                recipe_id, title, name, views,
                intro, servings, difficulty, time_min,
                full_text, image_url,
                method, situation, categories,
                main_ingredients, ingredients
            )

if __name__ == "__main__":
    csv_path = "dataset_preprocessed.csv"  # 너희 전처리된 csv 경로
    build_graph_from_csv(csv_path)
    print("✅ Graph build finished.")