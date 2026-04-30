import re
import math
import time
import random
import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from new_extractor_model import extract_keywords

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def normalize_basic(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
    return t.lower()

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        if not x.strip():
            return []
        return [x]
    return list(x)


def canonicalize_ingredient_list(lst):
    out = []
    for s in lst:
        if not s:
            continue
        s = s.strip().lower()
        if not s:
            continue
        out.append(s)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def softmax(scores, temperature: float = 1.0):
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp((s - max_s) / temperature) for s in scores]
    Z = sum(exps)
    if Z == 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / Z for e in exps]

DIFFICULTY_MAP = {
    "쉬운": ["아무나", "초급"],
    "간단": ["아무나", "초급"],
    "쉽": ["아무나", "초급"],
    "초보": ["아무나", "초급"],
    "입문": ["아무나", "초급"],
    "초급": ["초급"],
    "중급": ["중급"],
    "고급": ["고급"],
    "어려운": ["고급", "중급"],
    "힘든": ["고급", "중급"],
}

def normalize_difficulty(raw_kw):
    diffs = raw_kw.get("difficulty", [])
    text = raw_kw.get("free_text", "").lower()
    detected = set()
    for d in diffs:
        d = d.lower()
        for key, mapped in DIFFICULTY_MAP.items():
            if key in d:
                detected.update(mapped)
    for key, mapped in DIFFICULTY_MAP.items():
        if key in text:
            detected.update(mapped)
    return list(detected)


def get_all_user_keywords(kw: dict):
    flat = []
    list_fields = [
        "must_ingredients", "optional_ingredients", "exclude_ingredients",
        "dish_type", "method", "situation", "health_tags", "weather_tags",
        "menu_style", "extra_keywords", "difficulty", "positive_tags", "negative_tags"
    ]
    for lf in list_fields:
        vals = kw.get(lf, [])
        if vals:
            flat.extend([v for v in vals if v])

    serv = kw.get("servings", {})
    if serv.get("min") is not None:
        flat.append(f"servings_min:{serv['min']}")
    if serv.get("max") is not None:
        flat.append(f"servings_max:{serv['max']}")

    dc = kw.get("dietary_constraints", {})
    for key, val in dc.items():
        if val is True:
            flat.append(f"diet:{key}")

    max_t = kw.get("max_cook_time_min")
    if max_t:
        flat.append(f"max_time:{max_t}")

    seen = set()
    out = []
    for x in flat:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_cypher_from_keywords_relaxed(kw: dict, filterKeywords: list = [], limit: int = 50):
    kw = dict(kw)
    ex = kw["exclude_ingredients"]
    for ing in ex:
        filterKeywords["include"] = list(filter(lambda x: x["name"] != ing, filterKeywords["include"]))

    for item in filterKeywords["include"]:
        name = item["name"]
        field = item["field"]
        state = item["state"]
        if state == "include":
            kw[field].append(name.strip())

    for item in filterKeywords["exclude"]:
        name = item["name"]
        field = item["field"]
        state = item["state"]
        if state == "exclude":
            exclude = True
            for k, value in kw.items():
                if type(value) == list and name in value:
                    exclude = False
                    break
            if exclude:
                kw[field].append(name.strip())

    list_keys = [
        "dish_type", "method", "situation",
        "must_ingredients", "optional_ingredients", "exclude_ingredients",
        "health_tags", "weather_tags", "menu_style",
        "extra_keywords", "positive_tags", "difficulty"
    ]
    for k in list_keys:
        kw[k] = ensure_list(kw.get(k))

    def unique_preserve(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    for k in ["dish_type", "method", "situation", "menu_style", "extra_keywords", "health_tags", "difficulty"]:
        kw[k] = unique_preserve(kw[k])

    for key in ["must_ingredients", "optional_ingredients", "exclude_ingredients"]:
        kw[key] = canonicalize_ingredient_list(kw[key]) if kw[key] else []

    dc = kw.get("dietary_constraints", {})
    vegetarian = bool(dc.get("vegetarian"))
    vegan      = bool(dc.get("vegan"))
    no_beef    = bool(dc.get("no_beef"))
    no_pork    = bool(dc.get("no_pork"))
    no_chicken = bool(dc.get("no_chicken"))
    no_seafood = bool(dc.get("no_seafood"))

    serv_min = kw.get("servings", {}).get("min")
    serv_max = kw.get("servings", {}).get("max")
    menu_names = kw.get("dish_type", []) + kw.get("extra_keywords", [])

    params = {
        "must_ings": kw["must_ingredients"],
        "opt_ings": kw["optional_ingredients"],
        "exclude_ings": kw["exclude_ingredients"],
        "dish_type": kw["dish_type"],
        "method_list": kw["method"],
        "situation_list": kw["situation"],
        "health_list": kw["health_tags"],
        "weather_list": kw["weather_tags"],
        "menu_style_list": kw["menu_style"],
        "extra_kw_list": kw["extra_keywords"],
        "difficulty_list": kw["difficulty"],
        "menu_name_list": menu_names,
        "serv_min": serv_min,
        "serv_max": serv_max,
        "vegetarian": vegetarian,
        "vegan": vegan,
        "no_beef": no_beef,
        "no_pork": no_pork,
        "no_chicken": no_chicken,
        "no_seafood": no_seafood,
        "max_time": kw.get("max_cook_time_min", None),
        "limit_number": limit,
    }

    cypher = """
MATCH (r:RecipeV2)
OPTIONAL MATCH (r)-[:HAS_INGREDIENT_V2]->(ing:IngredientV2)
OPTIONAL MATCH (r)-[:IN_CATEGORY_V2]->(cat:CategoryV2)
OPTIONAL MATCH (r)-[:COOKED_BY_V2]->(meth:MethodV2)
OPTIONAL MATCH (r)-[:FOR_SITUATION_V2]->(sit:SituationV2)
OPTIONAL MATCH (r)-[:HAS_HEALTH_TAG]->(h:HealthTag)
OPTIONAL MATCH (r)-[:HAS_WEATHER_TAG]->(w:WeatherTag)
OPTIONAL MATCH (r)-[:HAS_MENU_STYLE]->(ms:MenuStyle)
OPTIONAL MATCH (r)-[:HAS_EXTRA_KEYWORD]->(ek:ExtraKeyword)

WITH r,
     collect(DISTINCT ing.name) AS ingRaw,
     collect(DISTINCT cat.name) AS catRaw,
     collect(DISTINCT meth.name) AS methodRaw,
     collect(DISTINCT sit.name) AS sitRaw,
     collect(DISTINCT h.name) AS healthRaw,
     collect(DISTINCT w.name) AS weatherRaw,
     collect(DISTINCT ms.name) AS menuStyleRaw,
     collect(DISTINCT ek.name) AS extraRaw

WITH
    r,
    [x IN ingRaw | replace(toLower(x)," ","")] AS ingList,
    [x IN catRaw | replace(toLower(x)," ","")] AS catList,
    [x IN methodRaw | replace(toLower(x)," ","")] AS methodList,
    [x IN sitRaw | replace(toLower(x)," ","")] AS sitList,
    [x IN healthRaw | replace(toLower(x)," ","")] AS healthList,
    [x IN weatherRaw | replace(toLower(x)," ","")] AS weatherList,
    [x IN menuStyleRaw | replace(toLower(x)," ","")] AS menuStyleList,
    [x IN extraRaw | replace(toLower(x)," ","")] AS extraList,
    r.servings AS servings_num

WHERE (
    size($must_ings) = 0 OR
    ALL(ing IN $must_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ing)," ","")))
)
AND (
    size($exclude_ings) = 0 OR
    NONE(ex IN $exclude_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ex)," ","")))
)
AND ($max_time IS NULL OR r.time_min <= $max_time)
AND ($serv_min IS NULL OR servings_num >= $serv_min)
AND ($serv_max IS NULL OR servings_num <= $serv_max)
AND (NOT $vegetarian OR NOT ANY(mi IN ingList WHERE mi CONTAINS '소고기' OR mi CONTAINS '돼지고기' OR mi CONTAINS '닭고기' OR mi CONTAINS '해산물'))
AND (NOT $vegan OR NOT ANY(mi IN ingList WHERE mi CONTAINS '소고기' OR mi CONTAINS '돼지고기' OR mi CONTAINS '닭고기' OR mi CONTAINS '해산물' OR mi CONTAINS '계란' OR mi CONTAINS '우유'))
AND (NOT $no_beef OR NOT ANY(mi IN ingList WHERE mi CONTAINS '소고기'))
AND (NOT $no_pork OR NOT ANY(mi IN ingList WHERE mi CONTAINS '돼지고기'))
AND (NOT $no_chicken OR NOT ANY(mi IN ingList WHERE mi CONTAINS '닭고기'))
AND (NOT $no_seafood OR NOT ANY(mi IN ingList WHERE mi CONTAINS '해산물'))

WITH
    r, servings_num,
    ingList, catList, methodList, sitList, healthList, weatherList, menuStyleList, extraList,
    size([ing IN $must_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ing)," ",""))]) * 5 AS score_must_ing,
    size([ing IN $opt_ings WHERE ANY(mi IN ingList WHERE mi CONTAINS replace(toLower(ing)," ",""))]) * 2 AS score_opt_ing,
    size([dt IN $dish_type WHERE ANY(cat IN catList WHERE cat CONTAINS replace(toLower(dt)," ",""))]) * 3 AS score_dish_type,
    size([mt IN $method_list WHERE ANY(m IN methodList WHERE m CONTAINS replace(toLower(mt)," ",""))]) * 2 AS score_method,
    size([st IN $situation_list WHERE ANY(s IN sitList WHERE s CONTAINS replace(toLower(st)," ",""))]) * 4 AS score_situation,
    size([ht IN $health_list WHERE ANY(h IN healthList WHERE h CONTAINS replace(toLower(ht)," ","") OR replace(toLower(ht)," ","") CONTAINS h)]) * 5 AS score_health,
    size([wt IN $weather_list WHERE ANY(w IN weatherList WHERE w CONTAINS replace(toLower(wt)," ",""))]) * 3 AS score_weather,
    size([ms IN $menu_style_list WHERE ANY(m IN menuStyleList WHERE m CONTAINS replace(toLower(ms)," ",""))]) * 2 AS score_menu_style,
    size([ek IN $extra_kw_list WHERE ANY(e IN extraList WHERE e CONTAINS replace(toLower(ek)," ","") OR replace(toLower(ek)," ","") CONTAINS e)]) * 3 AS score_extra,
    size([df IN $difficulty_list WHERE toLower(r.difficulty) CONTAINS replace(toLower(df)," ","")]) * 4 AS score_difficulty,
    size([mn IN $menu_name_list WHERE toLower(r.name) CONTAINS replace(toLower(mn)," ","") OR toLower(r.title) CONTAINS replace(toLower(mn)," ","")]) * 10 AS score_menu_name,
    CASE
        WHEN $serv_min IS NULL THEN 0
        WHEN servings_num IS NULL THEN 0
        WHEN servings_num = $serv_min THEN 5
        WHEN abs(servings_num - $serv_min) = 1 THEN 3
        ELSE 0
    END AS score_servings

WITH
    r,
    score_must_ing, score_opt_ing, score_dish_type, score_method, score_situation,
    score_health, score_weather, score_menu_style, score_extra,
    score_difficulty, score_menu_name, score_servings,
    (score_must_ing + score_opt_ing + score_dish_type + score_method + score_situation +
     score_health + score_weather + score_menu_style + score_extra +
     score_difficulty + score_menu_name + score_servings) AS score

RETURN
    r.recipe_id AS recipe_id, r.title AS title, r.name AS name, r.views AS views,
    r.time_min AS time_min, r.difficulty AS difficulty, r.servings AS servings,
    r.image_url AS image_url,
    score, score_must_ing, score_opt_ing, score_dish_type, score_method, score_situation,
    score_health, score_weather, score_menu_style, score_extra,
    score_difficulty, score_menu_name, score_servings
ORDER BY score DESC, r.views DESC
LIMIT $limit_number
"""
    return cypher, params, kw


def _norm_tag(s: str) -> str:
    return str(s).replace(" ", "").lower()


def _build_match_dict(llm_list, graph_list):
    result = {}
    norm_graph = [(g, _norm_tag(g)) for g in graph_list]
    for kw in llm_list:
        norm_kw = _norm_tag(kw)
        if not norm_kw:
            continue
        hits = [g for (g, ng) in norm_graph if norm_kw in ng or ng in norm_kw]
        if hits:
            result[kw] = hits
    return result


def graph_rag_search_with_scoring_explanation(
    user_prompt: str,
    top_k: int = 5,
    greedy_k: int = 3,
    filterKeywords: dict = {},
    temperature: float = 1.5,
):
    print("\n" + "=" * 80)
    print("USER PROMPT:", user_prompt)

    raw_kw = extract_keywords(user_prompt)
    raw_kw["difficulty"] = normalize_difficulty(raw_kw)

    cypher, params, kw = build_cypher_from_keywords_relaxed(raw_kw, filterKeywords=filterKeywords, limit=50)
    matched_keywords_only = get_all_user_keywords(raw_kw)

    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = list(result)

    if not rows:
        return {"keywords": kw, "recipes": []}

    all_zero = all((rec["score"] or 0) == 0 for rec in rows)
    if all_zero:
        return {
            "keywords": kw,
            "recipes": [],
            "no_result_message": "조회 가능한 메뉴가 없습니다. 프롬프트를 조금 더 구체적으로 입력해 주세요.",
        }

    top_candidates = rows

    if len(top_candidates) <= top_k:
        selected_rows = top_candidates
    else:
        greedy_k = min(greedy_k, top_k, len(top_candidates))
        cutoff_score = top_candidates[greedy_k - 1]["score"]
        tied = [rec for rec in top_candidates if rec["score"] == cutoff_score]
        greedy_zone = top_candidates[:greedy_k]
        tied_in_greedy_zone = [rec for rec in greedy_zone if rec["score"] == cutoff_score]

        if len(tied_in_greedy_zone) > 1:
            greedy_part = random.sample(tied, greedy_k)
        else:
            greedy_part = greedy_zone

        diversity_needed = top_k - greedy_k
        diversity_pool = [rec for rec in top_candidates if rec not in greedy_part]

        if diversity_needed <= 0 or not diversity_pool:
            selected_rows = greedy_part
        else:
            scores = [rec["score"] for rec in diversity_pool]
            probs = softmax(scores, temperature=temperature)
            chosen_idx = []
            while len(chosen_idx) < diversity_needed and len(chosen_idx) < len(diversity_pool):
                r = random.random()
                cum = 0.0
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        if i not in chosen_idx:
                            chosen_idx.append(i)
                        break

            diverse_part = [diversity_pool[i] for i in chosen_idx]
            selected_rows = greedy_part + diverse_part

            unique_rows = []
            seen_names = set()
            for rec in selected_rows:
                norm_name = rec["name"].replace(" ", "").lower()
                if norm_name not in seen_names:
                    seen_names.add(norm_name)
                    unique_rows.append(rec)

            if len(unique_rows) < top_k:
                for rec in top_candidates:
                    norm_name = rec["name"].replace(" ", "").lower()
                    if norm_name not in seen_names:
                        seen_names.add(norm_name)
                        unique_rows.append(rec)
                        if len(unique_rows) == top_k:
                            break

            selected_rows = unique_rows[:top_k]

    recipe_detail_query = """
    MATCH (r:RecipeV2 {recipe_id: $rid})
    OPTIONAL MATCH (r)-[:HAS_HEALTH_TAG]->(h:HealthTag)
    OPTIONAL MATCH (r)-[:HAS_WEATHER_TAG]->(w:WeatherTag)
    OPTIONAL MATCH (r)-[:HAS_MENU_STYLE]->(ms:MenuStyle)
    OPTIONAL MATCH (r)-[:HAS_EXTRA_KEYWORD]->(ek:ExtraKeyword)
    OPTIONAL MATCH (r)-[:FOR_SITUATION_V2]->(sit:SituationV2)
    OPTIONAL MATCH (r)-[:COOKED_BY_V2]->(meth:MethodV2)
    OPTIONAL MATCH (r)-[:IN_CATEGORY_V2]->(cat:CategoryV2)
    RETURN
      collect(DISTINCT h.name) AS healthList, collect(DISTINCT w.name) AS weatherList,
      collect(DISTINCT ms.name) AS menuStyleList, collect(DISTINCT ek.name) AS extraList,
      collect(DISTINCT sit.name) AS situationList, collect(DISTINCT meth.name) AS methodList,
      collect(DISTINCT cat.name) AS categoryList
    """

    recipes = []
    with driver.session() as session:
        for rec in selected_rows:
            r_info = {
                "recipe_id": rec["recipe_id"],
                "title": rec["title"],
                "name": rec["name"],
                "views": rec["views"],
                "time_min": rec["time_min"],
                "difficulty": rec["difficulty"],
                "servings": rec.get("servings"),
                "score": rec["score"],
                "score_must_ing": rec["score_must_ing"],
                "score_opt_ing": rec["score_opt_ing"],
                "score_dish_type": rec["score_dish_type"],
                "score_method": rec["score_method"],
                "score_situation": rec["score_situation"],
                "score_health": rec["score_health"],
                "score_weather": rec["score_weather"],
                "score_menu_style": rec["score_menu_style"],
                "score_extra": rec["score_extra"],
                "score_servings": rec.get("score_servings", 0),
                "score_difficulty": rec["score_difficulty"],
                "score_menu_name": rec["score_menu_name"],
                "image_url": rec["image_url"],
            }

            detail = session.run(recipe_detail_query, {"rid": rec["recipe_id"]}).single()

            user_keywords_all = (
                kw.get("must_ingredients", []) + kw.get("optional_ingredients", []) +
                kw.get("dish_type", []) + kw.get("method", []) + kw.get("situation", []) +
                kw.get("health_tags", []) + kw.get("weather_tags", []) +
                kw.get("menu_style", []) + kw.get("extra_keywords", [])
            )
            seen_kw = set()
            flat_unique = []
            for k in user_keywords_all:
                if k not in seen_kw:
                    seen_kw.add(k)
                    flat_unique.append(k)

            r_info["matched_keywords_flat"] = flat_unique
            recipes.append(r_info)

    return {"keywords": kw, "recipes": recipes}
