import React, { useState, useEffect } from "react";
import { useSession } from "../../Context/SessionContext";
import formatKeywords from "./formatKeywords";
import classes from "./RecipeSearchPage.module.css";

import ResultList from "./ResultList";
import Spinner from "../../Assets/Components/Spinner";

function RecipeSearchPage() {
  const {
    sessionId,
    currentPrompt,
    setCurrentPrompt,
    addInteraction,
    matchedKeywords,
    recommendations,
  } = useSession();

  const [loading, setLoading] = useState(false);
  const [filterKeywords, setFilterKeywords] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    setFilterKeywords(formatKeywords(matchedKeywords));
  }, [matchedKeywords]);

  useEffect(() => {
    if (!recommendations || recommendations.length === 0) return;

    window.scrollTo({
      top: 0,
      behavior: "smooth", // 부드럽게 스크롤
    });
  }, [recommendations]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const trimmed = currentPrompt.trim();
    if (!trimmed) {
      setError("프롬프트를 입력해주세요.");
      return;
    }

    setLoading(true);

    try {
      // const res = await fetch("http://localhost:5000/api/recipe-search", {
      const res = await fetch(
        "http://localhost:5000/api/jiewan-recipe-search",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            sessionId: sessionId,
            query: trimmed,
            matchedKeywords: matchedKeywords,
            filterKeywords: filterKeywords,
            top_k: 5,
          }),
        }
      );

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "서버 오류");
      }
      setCurrentPrompt("");
      const data = await res.json();

      addInteraction({
        prompt: trimmed,
        matchedKeywords: data.matchedKeywords || [],
        recommendations: data.recommendations || [],
      });
    } catch (err) {
      console.error(err);
      setError(err.message || "요청 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={classes.container}>
      <div className={classes.contentBox}>
        {recommendations.length > 0 && <ResultList results={recommendations} />}
        <div className={classes.promptContainer}>
          <div className={classes.titleContainer}>
            <div className={classes.title}>
              {recommendations.length > 0
                ? "추가로 원하시는 조건을 입력해주세요!"
                : "AI한테 맡겨보세요!!"}
            </div>
            <div className={classes.subtitle}>
              {true ? "식재료와 취향을 입력해주세요." : "AI 추천 문장"}
            </div>
          </div>
          <div className={classes.keywordsContainer}>
            {!filterKeywords ||
            !filterKeywords.include ||
            !filterKeywords.exclude ? null : filterKeywords.include.length +
                filterKeywords.exclude.length >
              0 ? (
              <div className={classes.keywordsContainer}>
                <div className={classes.keywordsTitle}>매칭된 키워드</div>
                <div className={classes.keywordsBox}>
                  {filterKeywords.include.length > 0 ? (
                    <div className={classes.keywordsLineBox}>
                      {filterKeywords.include.map((item, idx) => {
                        return (
                          <div
                            key={idx}
                            className={`${classes.keywordItem}  ${
                              item.state === "ignore" ? classes.canceled : ""
                            }`}
                            onClick={() => {
                              setFilterKeywords((prev) => ({
                                ...prev,
                                include: prev.include.map((obj) =>
                                  item.name === obj.name
                                    ? {
                                        ...obj,
                                        state:
                                          obj.state === "ignore"
                                            ? "include"
                                            : "ignore",
                                      }
                                    : obj
                                ),
                              }));
                            }}
                          >
                            <div>{item.name}</div>
                            <div>x</div>
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                  {filterKeywords.exclude.length > 0 ? (
                    <div className={classes.keywordsLineBox}>
                      {filterKeywords.exclude.map((item, idx) => {
                        return (
                          <div
                            key={idx}
                            className={`${classes.keywordItem}  ${
                              classes.keywordItemExclude
                            } ${
                              item.state === "ignore" ? classes.canceled : ""
                            }`}
                            onClick={() => {
                              setFilterKeywords((prev) => ({
                                ...prev,
                                exclude: prev.exclude.map((obj) =>
                                  item.name === obj.name
                                    ? {
                                        ...obj,
                                        state:
                                          obj.state === "ignore"
                                            ? "exclude"
                                            : "ignore",
                                      }
                                    : obj
                                ),
                              }));
                            }}
                          >
                            <div>{item.name}</div>
                            <div>x</div>
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>

          <form onSubmit={handleSubmit} className={classes.form}>
            <textarea
              rows={3}
              className={classes.textarea}
              placeholder={
                recommendations.length > 0
                  ? "추가로 원하시는 사항을 입력해주세요."
                  : "예: 떡이 들어가고 너무 맵지 않은 따뜻한 국물 요리 추천해줘"
              }
              value={currentPrompt}
              onChange={(e) => setCurrentPrompt(e.target.value)}
            />
            <button
              type="submit"
              disabled={loading}
              className={`${classes.button} ${
                loading ? classes.buttonDisabled : ""
              }`}
            >
              {loading ? (
                <>
                  <span style={{ display: "flex" }}> 검색중...</span>
                  {/* <Spinner /> */}
                </>
              ) : (
                "레시피 추천받기"
              )}
            </button>
          </form>
          {error && <div className={classes.error}>{error}</div>}

          {!loading && !error && recommendations.length === 0 && (
            <p className={classes.emptyText}>
              프롬프트를 입력하고 검색을 눌러보세요.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default RecipeSearchPage;
