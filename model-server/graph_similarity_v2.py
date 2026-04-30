# graph_similarity.py
from typing import List, Dict, Any
from neo4j import GraphDatabase
import math

# ================================
# 1. 유사도 & 다양성 헬퍼 함수
# ================================
def jaccard_similarity(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def diversify_by_set_field(
    candidates: List[Dict[str, Any]],
    field: str,
    top_n: int,
    lambda_rel: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    candidates: [{..., "score": float, field: List[str], ...}, ...]
    field     : "shared_ingredients" 또는 "shared_tags"
    top_n     : 최종 뽑을 개수
    lambda_rel: 기준 레시피와의 유사도 비중 (0~1, 클수록 정확도 위주)
    """
    if len(candidates) <= top_n:
        return candidates

    # score 기준 정렬 (혹시 정렬 안 되어 있을 수 있으니)
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []
    remaining = candidates.copy()

    # 첫 번째는 무조건 최고 점수
    first = remaining.pop(0)
    selected.append(first)

    # 나머지 top_n-1개를 MMR 기준으로 선택
    while len(selected) < top_n and remaining:
        best_item = None
        best_mmr = -math.inf

        for cand in remaining:
            rel = cand["score"]
            # 이미 선택된 것들과의 최대 유사도 (겹치는 정도)
            max_sim = 0.0
            for s in selected:
                sim = jaccard_similarity(
                    cand.get(field, []),
                    s.get(field, []),
                )
                if sim > max_sim:
                    max_sim = sim

            # MMR 점수
            mmr = lambda_rel * rel - (1.0 - lambda_rel) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_item = cand

        selected.append(best_item)
        remaining.remove(best_item)

    return selected


# ================================
# 2. 메인 클래스
# ================================
class RecipeGraphSimilarity:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_similar_recipes(
        self,
        recipe_id: int,
        top_n: int = 3,
        min_shared_ings: int = 2,
        lambda_ing: float = 0.7,
        lambda_overall: float = 0.7,
        candidate_factor: int = 5,
    ):
        """
        - 재료 기준: 상위 candidate_n개를 Neo4j에서 가져온 뒤,
          MMR 기반으로 서로 다른 재료를 공유하도록 top_n개 선택
        - overall 기준: 재료로 이미 선택한 recipe_id는 제외하고
          태그 기반 후보 candidate_n개 가져와서,
          shared_tags 기준으로 다양성 있게 top_n개 선택
        """
        candidate_n = top_n * candidate_factor

        with self.driver.session() as session:
            # 1) 재료 기반 후보 넉넉히 가져오기
            ing_candidates = session.execute_read(
                self._query_ingredient_similar,
                recipe_id,
                candidate_n,
                min_shared_ings,
            )

            # 1-1) 재료 기반 diversified top_n
            ingredients = diversify_by_set_field(
                candidates=ing_candidates,
                field="shared_ingredients",
                top_n=top_n,
                lambda_rel=lambda_ing,
            )

            # 2) 재료 기반으로 이미 뽑힌 recipe_id 리스트
            exclude_ids = [row["recipe_id"] for row in ingredients]

            # 3) overall 후보 넉넉히 가져오기 (재료 기반 제외)
            overall_candidates = session.execute_read(
                self._query_overall_similar,
                recipe_id,
                candidate_n,
                exclude_ids,
            )

            # 3-1) overall에서도 shared_tags 기준 diversified top_n
            overall = diversify_by_set_field(
                candidates=overall_candidates,
                field="shared_tags",
                top_n=top_n,
                lambda_rel=lambda_overall,
            )

        return {
            "overall": overall,
            "ingredients": ingredients,
        }

    # ==========================================================
    # 1) 전체 그래프 기준 – 관계 타입별 가중치 (재료 제외) + exclude_ids 필터
    #    ★ 여기서 LIMIT $candidate_n 유지 + exclude_ids 그대로 사용
    # ==========================================================
    @staticmethod
    def _query_overall_similar(tx, recipe_id, candidate_n, exclude_ids):
        cypher = """
        MATCH (base:RecipeV2 {recipe_id: $recipe_id})

        // base와 연결된 모든 태그 노드 (재료 제외)
        MATCH (base)-[:IN_CATEGORY_V2
                      |COOKED_BY_V2
                      |FOR_SITUATION_V2
                      |HAS_HEALTH_TAG
                      |HAS_WEATHER_TAG
                      |HAS_MENU_STYLE
                      |HAS_EXTRA_KEYWORD]->(t)

        // 같은 태그 t에 연결된 other 레시피
        MATCH (other:RecipeV2)-[r2:IN_CATEGORY_V2
                                |COOKED_BY_V2
                                |FOR_SITUATION_V2
                                |HAS_HEALTH_TAG
                                |HAS_WEATHER_TAG
                                |HAS_MENU_STYLE
                                |HAS_EXTRA_KEYWORD]->(t)
        WHERE other <> base
          AND NOT other.recipe_id IN $exclude_ids   // 재료 기반으로 뽑힌 것들 제외

        WITH other, t, type(r2) AS rel_type

        // 관계 타입별 가중치 합산
        WITH other,
             collect(DISTINCT t.name) AS shared_tags,
             sum(
               CASE rel_type
                 WHEN "FOR_SITUATION_V2"  THEN 4
                 WHEN "HAS_HEALTH_TAG"    THEN 5
                 WHEN "IN_CATEGORY_V2"    THEN 2
                 WHEN "HAS_WEATHER_TAG"   THEN 2
                 WHEN "HAS_MENU_STYLE"    THEN 2
                 WHEN "HAS_EXTRA_KEYWORD" THEN 3
                 ELSE 1
               END
             ) AS similarity_score

        RETURN
            other.recipe_id AS recipe_id,
            other.title     AS title,
            other.name      AS name,
            other.image_url AS image_url,
            similarity_score AS score,
            shared_tags
        ORDER BY score DESC, other.views DESC, title ASC
        LIMIT $candidate_n;
        """

        result = tx.run(
            cypher,
            recipe_id=recipe_id,
            candidate_n=candidate_n,
            exclude_ids=exclude_ids,
        )
        return [dict(record) for record in result]

    # ==========================================================
    # 2) 재료만 기준 – 공유 재료 수 기반 + 최소 재료 공유 필터
    #    ★ 여기서도 LIMIT $candidate_n으로 후보를 넉넉히
    # ==========================================================
    @staticmethod
    def _query_ingredient_similar(tx, recipe_id, candidate_n, min_shared_ings):
        cypher = """
        MATCH (base:RecipeV2 {recipe_id: $recipe_id})
        MATCH (base)-[:HAS_INGREDIENT_V2]->(ing:IngredientV2)

        MATCH (other:RecipeV2)-[:HAS_INGREDIENT_V2]->(ing)
        WHERE other <> base

        WITH other,
             collect(DISTINCT ing.name) AS shared_ingredients,
             count(DISTINCT ing)        AS shared_ing_count

        // 재료 공유 개수 필터
        WHERE shared_ing_count >= $min_shared_ings

        RETURN
            other.recipe_id       AS recipe_id,
            other.title           AS title,
            other.name            AS name,
            other.image_url       AS image_url,
            shared_ing_count      AS score,
            shared_ingredients
        ORDER BY score DESC, other.views DESC, title ASC
        LIMIT $candidate_n;
        """

        result = tx.run(
            cypher,
            recipe_id=recipe_id,
            candidate_n=candidate_n,
            min_shared_ings=min_shared_ings,
        )
        return [dict(record) for record in result]