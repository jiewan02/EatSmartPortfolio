from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Set MODEL_PATH in model-server/.env to the absolute path of your Qwen2.5-14B-Instruct model.
# Example: MODEL_PATH=/absolute/path/to/Qwen2.5-14B-Instruct
MODEL_NAME = os.getenv("MODEL_PATH")
if not MODEL_NAME:
    raise ValueError("MODEL_PATH is not set. Add it to model-server/.env pointing to your local Qwen2.5-14B-Instruct model.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

max_memory = {
    0: "20GiB",
    1: "20GiB",
    2: "20GiB",
    "cpu": "32GiB",
}

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (this can take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 2},
    max_memory=max_memory,
)

if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
    model.config.eos_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id


SYSTEM_PROMPT = """
당신은 한국어 레시피 추천 시스템의 핵심 구성요소인
"요리 의도·조건 추출기(Keyword & Constraint Extractor)" 입니다.

당신의 임무는:
사용자의 자연어 요리 요청에서 **모든 의미 있는 요소를 빠짐없이 구조화된 JSON으로 추출하는 것**입니다.

출력 규칙은 반드시 다음을 따라야 합니다:

────────────────────────────────────────
📌 **① 출력 형식**

오직 아래 JSON 스키마 형태의 JSON만 출력하십시오.
문장 설명, 해설, 추가 문구는 절대 출력하지 마십시오.
무조건 답변은 한국어로 하십시오.

모든 필드는 반드시 존재해야 합니다.
값이 비었으면 **빈 배열 또는 null** 로 채우십시오.

JSON 스키마:

{
  "dish_type": [],
  "method": [],
  "situation": [],
  "must_ingredients": [],
  "optional_ingredients": [],
  "exclude_ingredients": [],
  "spiciness": "none" | "low" | "medium" | "high" | null,
  "dietary_constraints": {
    "vegetarian": bool,
    "vegan": bool,
    "no_beef": bool,
    "no_pork": bool,
    "no_chicken": bool,
    "no_seafood": bool
  },
  "servings": { "min": int or null, "max": int or null },
  "max_cook_time_min": int or null,
  "difficulty": [],
  "health_tags": [],
  "weather_tags": [],
  "menu_style": [],
  "extra_keywords": [],
  "positive_tags": [],
  "negative_tags": [],
  "free_text": string
}

────────────────────────────────────────
📌 **② 각 필드의 세부 지침 (매우 중요)**

1) dish_type: "찌개", "국", "볶음", "튀김", "조림", "덮밥", "비빔밥", "면 요리" 등
2) method: "끓이기", "볶기", "찜", "무침", "튀기기", "굽기" 등
3) situation: "야식", "혼밥", "술안주", "손님 초대", "간단하게", "도시락", "캠핑" 등
4) must_ingredients: 사용자가 특정 재료를 언급하며 해당 요리를 원할 때
5) optional_ingredients: "있으면 좋고", "가능하면", "추가로" 표현된 재료
6) exclude_ingredients: "싫어", "알레르기", "못먹어", "빼고", "제외해줘" 등
7) spiciness: none | low | medium | high
8) dietary_constraints: 채식/비건/특정 육류 제한 여부
9) servings: 인분수 요구가 있을 때만 기입, 없으면 null
10) max_cook_time_min: 숫자가 명확할 때만, 없으면 null
11) difficulty: 난이도 표현 그대로
12) health_tags: "다이어트", "고단백", "저염식" 등
13) weather_tags: "추운 날", "더운 날", "비오는 날" 등
14) menu_style: "한식", "중식", "양식", "분식" 등
15) extra_keywords: 위 어디에도 속하지 않는 의미 있는 단어
16) positive_tags: 사용자가 원함/좋아함 표현
17) negative_tags: 사용자가 싫어함/피하고 싶음 표현
18) free_text: 전체 요청의 자연어 요약 (1~2문장)

────────────────────────────────────────
📌 **③ 절대적으로 지켜야 할 3가지**

1. 출력은 반드시 **JSON 단독**이어야 함
2. JSON 스키마의 **모든 필드**를 반드시 출력
3. 빈 값도 반드시 포함 (절대 필드 누락 금지)
"""


def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        if not x.strip():
            return []
        return [x]
    return list(x)

def _unique_preserve_order(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _dedup_by_norm_space_lower(tags):
    norm_seen = set()
    out = []
    for t in _ensure_list(tags):
        norm = str(t).replace(" ", "").lower()
        if norm in norm_seen:
            continue
        norm_seen.add(norm)
        out.append(t)
    return out


def _postprocess_text_to_json(output_text: str, fallback_prompt: str) -> dict:
    output_text = output_text.strip()
    if output_text.startswith("```"):
        output_text = output_text.strip("`").strip()
        if output_text.startswith("json"):
            output_text = output_text[4:].strip()

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError:
        data = {}

    base = {
        "dish_type": [], "method": [], "situation": [],
        "must_ingredients": [], "optional_ingredients": [], "exclude_ingredients": [],
        "spiciness": None,
        "dietary_constraints": {
            "vegetarian": False, "vegan": False, "no_beef": False,
            "no_pork": False, "no_chicken": False, "no_seafood": False,
        },
        "servings": {"min": None, "max": None},
        "max_cook_time_min": None,
        "difficulty": [],
        "health_tags": [], "weather_tags": [], "menu_style": [], "extra_keywords": [],
        "positive_tags": [], "negative_tags": [],
        "free_text": fallback_prompt,
    }

    if not isinstance(data, dict):
        data = {}

    merged = base.copy()
    merged.update({k: v for k, v in data.items() if v is not None})

    list_fields = [
        "dish_type", "method", "situation",
        "must_ingredients", "optional_ingredients", "exclude_ingredients",
        "difficulty", "health_tags", "weather_tags", "menu_style", "extra_keywords",
        "positive_tags", "negative_tags",
    ]
    for k in list_fields:
        merged[k] = _ensure_list(merged.get(k))

    dc = merged.get("dietary_constraints") or {}
    merged["dietary_constraints"] = {
        "vegetarian": bool(dc.get("vegetarian", False)),
        "vegan": bool(dc.get("vegan", False)),
        "no_beef": bool(dc.get("no_beef", False)),
        "no_pork": bool(dc.get("no_pork", False)),
        "no_chicken": bool(dc.get("no_chicken", False)),
        "no_seafood": bool(dc.get("no_seafood", False)),
    }

    serv = merged.get("servings") or {}
    merged["servings"] = {"min": serv.get("min"), "max": serv.get("max")}

    valid_sp = {"none", "low", "medium", "high", None}
    sp = merged.get("spiciness")
    if isinstance(sp, str):
        sp = sp.lower().strip()
        if sp not in valid_sp:
            sp = None
    elif sp not in valid_sp:
        sp = None
    merged["spiciness"] = sp

    if not isinstance(merged.get("free_text"), str) or not merged["free_text"].strip():
        merged["free_text"] = fallback_prompt

    pos = merged.get("positive_tags", [])
    merged["health_tags"] = _unique_preserve_order(merged["health_tags"] + pos)
    merged["extra_keywords"] = _unique_preserve_order(merged["extra_keywords"] + pos)
    merged["weather_tags"] = _dedup_by_norm_space_lower(merged["weather_tags"])

    return merged


def extract_keywords(user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return _postprocess_text_to_json(output_text, fallback_prompt=user_prompt)


__all__ = ["extract_keywords", "tokenizer", "model"]
