import json
import time
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os

# =========================
# 0. 설정
# =========================

INPUT_CSV = "dataset_part2.csv"         # 네 원본 csv 경로
OUTPUT_CSV = "recipes_with_tags.csv"    # 최종 결과 csv 경로
PARTIAL_META_CSV = "recipes_with_tags_meta_partial.csv"  # 중간 메타 결과 저장
MODEL_NAME = "gpt-4o-mini"              # 필요하면 gpt-5.1, gpt-5.1-mini 등으로 변경
BATCH_SIZE = 20                         # 한 번에 보낼 레시피 개수 (비용/속도 따라 조정)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# 1. 프롬프트 생성 함수
# =========================
def build_batch_input(batch_df: pd.DataFrame) -> str:
    """
    한 batch의 레시피 정보를 JSON 리스트로 만들어 모델 input 텍스트로 사용.
    """
    records: List[Dict] = []
    for _, row in batch_df.iterrows():
        record = {
            "recipe_id": int(row["레시피일련번호"]),
            "title": str(row.get("레시피제목", "")),
            "name": str(row.get("요리명", "")),
            "views": int(row.get("조회수", 0)) if pd.notna(row.get("조회수", None)) else 0,
            "cook_method": str(row.get("요리방법설명", "")),     # 끓이기, 볶기 등
            "situation": str(row.get("요리상황설명", "")),      # 명절, 술안주, 집들이 등
            "type": str(row.get("요리종류별명", "")),           # ['국','탕'] 이런 정보
            "intro": str(row.get("요리소개_clean", row.get("요리소개", ""))),
            "ingredients_text": str(row.get("요리재료내용", "")), # [재료] ~ 형식
            "servings": str(row.get("요리인분명", "")),         # 2인분 등
            "difficulty": str(row.get("요리난이도명", "")),      # 초급, 중급 등
            "time": str(row.get("요리시간명", "")),             # 30분 이내, 2시간이내 등
        }
        records.append(record)

    return json.dumps(records, ensure_ascii=False, indent=2)


INSTRUCTIONS = """
너는 한국 요리 레시피 메타데이터 태깅 전문가다.

입력으로 여러 개의 레시피가 JSON 리스트 형태로 주어진다.
각 레시피에 대해 아래 정보를 추출하여 **반드시 JSON 리스트**로 반환해라.

각 원소는 다음 형태의 JSON 객체여야 한다:

{
  "recipe_id": <int>,                 // 입력에서 받은 레시피일련번호
  "health_tags": [                    // 건강/영양 관련 태그 (0개 이상)
    "다이어트", "고칼로리", "저칼로리", "저염식", "고단백", "저탄수",
    "채식", "비건", "저지방", "고섬유질", "키즈메뉴", "술안주"
  중에서 해당하는 것만 선택],
  "weather_tags": [                   // 날씨/계절 태그 (0개 이상)
    "추운날", "더운날", "비오는날", "눈오는날",
    "복날", "겨울", "여름", "봄", "가을", "장마철"
  중에서 해당하는 것만 선택],
  "menu_style": [                     // 메뉴 스타일/카테고리 (0개 이상)
    "한식", "중식", "일식", "양식", "퓨전",
    "분식", "디저트", "야식", "안주", "간식", "도시락"
  중에서 해당하는 것만 선택],
  "extra_keywords": [                 // 요리소개, 상황, 재료에서 추가로 뽑을 핵심 키워드 (자유 형식, 0~10개)
    "명절음식", "새해떡국", "따뜻한국물", ...
  ]
}

주의사항:
- 반드시 JSON 배열만 출력해라. 설명 문장, 주석, 코드블록 표시, 한국어 설명 등을 절대 섞지 마라.
- 각 recipe_id는 입력의 recipe_id와 정확히 일치해야 한다.
- 태그 리스트 안의 문자열은 한국어로만 작성해라.
- 해당하지 않는 태그는 포함하지 말고, 완전 모를 때는 빈 리스트[]로 둔다.
"""


# =========================
# 1-1. OpenAI 호출 + 재시도 로직
# =========================
def call_openai_for_batch(batch_df: pd.DataFrame) -> List[Dict]:
    batch_input = build_batch_input(batch_df)

    # 재시도 횟수 늘림: 6회
    max_attempts = 10
    last_exception = None

    for attempt in range(max_attempts):
        raw_text = ""  # 예외 발생 시에도 변수 존재하도록 초기화
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                instructions=INSTRUCTIONS,
                input=batch_input,
                max_output_tokens=2048,
            )

            # Responses API output 파싱
            raw_text_parts = []
            for out in response.output:
                for content in out.content:
                    if content.type == "output_text":
                        raw_text_parts.append(content.text)
            raw_text = "".join(raw_text_parts)

            if not raw_text.strip():
                raise ValueError(f"[API ERROR] 빈 응답. response={response}")

            # JSON 파싱
            data = json.loads(raw_text)
            return data

        except Exception as e:
            last_exception = e
            print(f"[WARN] OpenAI 호출 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if raw_text:
                print("raw_text (앞부분) =", raw_text[:200])
            # exponential backoff
            sleep_sec = 3 * (attempt + 1)
            print(f"→ {sleep_sec}초 대기 후 재시도")
            time.sleep(sleep_sec)

    # 여기까지 왔다면 모든 시도가 실패
    print("[ERROR] OpenAI API 호출이 모든 재시도에서 실패했습니다.")
    raise RuntimeError(f"OpenAI API 실패: {last_exception}")


# =========================
# 1-2. 중간 결과 저장 함수
# =========================
def save_partial_results(all_results: List[Dict], filename: str = PARTIAL_META_CSV):
    """
    지금까지 수집한 all_results를 meta-only CSV로 중간 저장.
    세션이 끊겨도 이 파일은 남아 있게 됨.
    """
    if not all_results:
        return

    meta_df = pd.DataFrame(all_results)

    # 리스트 컬럼들을 CSV에서 다루기 쉽게 문자열로 변환 (예: "태그1|태그2")
    for col in ["health_tags", "weather_tags", "menu_style", "extra_keywords"]:
        if col in meta_df.columns:
            meta_df[col] = meta_df[col].apply(
                lambda x: "|".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "")
            )

    meta_df.to_csv(filename, index=False)
    print(f"💾 중간 저장 완료: {filename} (rows={len(meta_df)})")


# =========================
# 2. 메인 로직
# =========================
def main():
    # 원본 데이터 읽기
    df = pd.read_csv(INPUT_CSV)

    all_results: List[Dict] = []

    # 배치 단위로 순회
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing batches"):
        end = min(start + BATCH_SIZE, len(df))
        batch_df = df.iloc[start:end]
        print(f"\n=== 배치 처리: rows {start} ~ {end-1} ===")

        try:
            batch_results = call_openai_for_batch(batch_df)
        except Exception as e:
            # 이 배치에서 총 실패했지만, 지금까지의 결과는 중간 저장
            print(f"[ERROR] 배치 {start}~{end-1} 처리 중 오류, 이 배치는 스킵합니다: {e}")
            save_partial_results(all_results)
            # 계속 진행할지, 여기서 멈출지 선택 가능
            # 여기서는 계속 진행 (다음 배치 시도)
            continue

        # 정상 응답을 all_results에 추가
        all_results.extend(batch_results)

        # 각 배치마다 중간 저장 (원하면 N배치마다로 바꿔도 됨)
        save_partial_results(all_results)

        # API 사용량이 너무 크지 않도록 약간의 sleep (필요에 따라 조정/삭제)
        time.sleep(0.5)

    # 최종 meta_df 생성
    if not all_results:
        print("[WARN] 수집된 결과가 없습니다. 프로그램을 종료합니다.")
        return

    meta_df = pd.DataFrame(all_results)

    # 리스트 → 문자열 변환
    for col in ["health_tags", "weather_tags", "menu_style", "extra_keywords"]:
        if col in meta_df.columns:
            meta_df[col] = meta_df[col].apply(
                lambda x: "|".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "")
            )

    # recipe_id 기준으로 원본 df와 머지
    meta_df.rename(columns={"recipe_id": "레시피일련번호"}, inplace=True)
    merged = df.merge(meta_df, on="레시피일련번호", how="left")

    # 최종 CSV 저장
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 완료! 최종 결과를 {OUTPUT_CSV} 로 저장했습니다.")


def test_api():
    resp = client.responses.create(
        model="gpt-4o-mini",
        input="API 연결이 잘 되면 이 문장을 그대로 출력해줘."
    )
    print(resp.output_text)


if __name__ == "__main__":
    # test_api()
    main()