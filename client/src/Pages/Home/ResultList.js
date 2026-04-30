import { useNavigate } from "react-router-dom";
import { useSession } from "../../Context/SessionContext";
import classes from "./ResultList.module.css";
import ResetButton from "../../Assets/Components/ResetButton";
import formatKeywords from "./formatKeywords";

const ResultList = ({ results }) => {
  const navigate = useNavigate();
  const { matchedKeywords } = useSession();

  console.log("results: ", results);
  console.log("k: ", formatKeywords(matchedKeywords));

  return (
    <div className={classes.resultsWrapper}>
      <div className={classes.titleContainer}>
        <div className={classes.resultsTitle}>
          검색 결과 (Top {results.length})
        </div>
        <ResetButton />
      </div>
      <ul className={classes.resultList}>
        {results.map((item, idx) => (
          <li
            key={item.index ?? idx}
            className={classes.resultItem}
            onClick={() => {
              navigate(`/recipe/${item.recipe_id}`, {
                state: {
                  data: item,
                },
              });
            }}
          >
            <div className={classes.itemImageCard}>
              <img
                className={classes.itemImage}
                src={item.image_url}
                alt="menu"
              />
            </div>
            <div className={classes.itemContent}>
              <div className={classes.itemHeader}>
                <strong>
                  {idx + 1}. {item.name || "이름 없음"}
                </strong>
                {/* <span className={classes.itemScore}>
                  {(item.score || 0).toFixed(3)}
                </span> */}
              </div>
              <div className={classes.metaRowSmall}>
                {item.servings && <>인분: {item.servings} </>}
                {item.difficulty && <>| 난이도: {item.difficulty} </>}
                {item.time && <>| 조리시간: {item.time}</>}
              </div>

              <div>
                매칭 점수:
                <span className={classes.itemScore}>
                  {(item.score || 0).toFixed(3)}
                </span>
              </div>
              <div className={classes.tagBox}>
                키워드{" "}
                {matchedKeywords &&
                  formatKeywords(matchedKeywords)
                    .include.filter((i) =>
                      item.matched_keywords_flat.includes(i.name)
                    )
                    .map((tag, idx) => {
                      return (
                        <div key={idx} className={classes.tag}>
                          {tag.name}
                        </div>
                      );
                    })}
              </div>

              {Array.isArray(item.types) && item.types.length > 0 && (
                <div className={classes.metaRow}>
                  종류: {item.types.join(", ")}
                </div>
              )}

              {Array.isArray(item.ingredients) &&
                item.ingredients.length > 0 && (
                  <div className={classes.metaRow}>
                    재료: {item.ingredients.slice(0, 10).join(", ")}
                    {item.ingredients.length > 10 && " ..."}
                  </div>
                )}

              {item.intro && (
                <div className={classes.metaRow}>소개: {item.intro}</div>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ResultList;
