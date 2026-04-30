import { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import classes from "./Recipe.module.css";
import reviewImage from "../../Assets/Images/review.png";
const Recipe = () => {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const data = location.state || {};
  const [steps, setSteps] = useState([]);
  const [, setImageUrl] = useState("");
  const [gridInfo, setGridInfo] = useState({});
  const [ingredients, setIngredients] = useState([]);
  const [overall, setOverall] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [title, setTitle] = useState("");
  const [infos, setInfos] = useState([]);
  const [activeState, setActiveState] = useState("ingredients");

  const [ingOpen, setIngOpen] = useState(false);
  const [stepsOpen, setStepsOpen] = useState(false);

  useEffect(() => {
    if (!id) return;

    const fetchRecipe = async () => {
      setLoading(true);
      setError("");
      setSteps([]);

      try {
        const res = await fetch(`http://localhost:5000/api/recipe/${id}`);
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || "Server error");
        }

        const data = await res.json();
        const { image_url, steps, grid_info, infos, title } = data.data;
        const { ingredients, overall } = data;
        // data: { id: number, steps: string[] }
        console.log("data", data);
        setTitle(title);
        setInfos(infos);
        setImageUrl(image_url || "");
        setGridInfo(grid_info || []);
        setSteps(steps || []);
        setIngredients(ingredients);
        setOverall(overall);
      } catch (err) {
        console.error(err);
        setError("레시피 정보를 불러오지 못했습니다.");
      } finally {
        setLoading(false);
      }
    };

    fetchRecipe();
  }, [id]);

  if (loading) return <div>불러오는 중...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className={classes.container}>
      <div className={classes.imageContainer}>
        <img
          className={classes.imageContainer}
          src={
            data.data.image_url ? data.data.image_url : data.data.data.image_url
          }
          alt=""
        />
      </div>
      <div className={classes.infoContainer}>
        <div className={classes.title}>
          {data.data.name ? data.data.name : title}
        </div>
        <div className={classes.recommend}>{data.data.title}</div>
        <div className={classes.data}>
          {infos[0] ? infos[0] : data.data.servings}인분 |{" "}
          {infos[1] ? infos[1] : data.data.difficulty} |{" "}
          {infos[2] ? infos[2] : ""}
        </div>
      </div>
      <div className={classes.contentContainer}>
        <div className={classes.toggleHeader}>
          <div
            className={`${classes.headerButton} ${
              activeState === "ingredients" ? classes.toggleActive : ""
            }`}
            onClick={() => {
              setActiveState("ingredients");
            }}
          >
            준비물
          </div>
          <div
            className={`${classes.headerButton} ${
              activeState === "steps" ? classes.toggleActive : ""
            }`}
            onClick={() => {
              setActiveState("steps");
            }}
          >
            조리순서
          </div>
        </div>

        <div>
          {activeState === "ingredients" ? (
            <div className={classes.ingredientsContainer}>
              <div
                className={classes.ingredientsColumn}
                style={ingOpen ? { marginTop: "30px" } : null}
              >
                {true &&
                  Object.keys(gridInfo).map((key, index) => {
                    return (
                      <div key={key} className={classes.ingredientsBox}>
                        <div
                          style={{ marginBottom: "20px", fontSize: "1.5rem" }}
                        >
                          {key}
                        </div>
                        {gridInfo[key].map((pair, i) => {
                          return (
                            <div key={i}>
                              {typeof pair === typeof [] ? (
                                <div className={classes.ingRow}>
                                  <div>{pair[0]}</div>
                                  {pair[1] ? <div>{pair[1]}</div> : null}
                                </div>
                              ) : (
                                <div>{pair}</div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}
              </div>
            </div>
          ) : steps.length === 0 ? (
            <p style={{ fontSize: "1.5rem", fontweight: "500" }}>
              단계 정보가 없습니다.
            </p>
          ) : (
            <div className={classes.stepsContainer}>
              <div className={classes.listContainer}>
                {true &&
                  steps.map((step, idx) => (
                    <div
                      key={idx}
                      style={{ lineHeight: "2rem" }}
                      className={classes.stepContainer}
                    >
                      <div className={classes.stepTextBox}>
                        <div className={classes.stepText}>{`${idx + 1}. ${
                          step.text
                        }`}</div>
                        <div className={classes.stepTool}>{step.tools}</div>
                      </div>
                      <div className={classes.stepImageBox}>
                        <img
                          className={classes.stepImage}
                          src={step.img_url}
                          alt=""
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className={classes.moreContainer}>
        <div className={classes.moreHeader}>이런 메뉴들은 어떠세요?</div>
        {ingredients.length > 0 ? (
          <div className={classes.moreBox}>
            <div className={classes.moreTitle}>비슷한 재료</div>
            <div className={classes.moreItemBox}>
              {ingredients.map((data, idx) => {
                return (
                  <div
                    key={idx}
                    className={classes.moreItem}
                    onClick={() => {
                      navigate(
                        `http://localhost:3000/recipe/${data.recipe_id}`,
                        {
                          state: {
                            data: { data: data },
                          },
                        }
                      );
                    }}
                  >
                    <div className={classes.moreCardImageContainer}>
                      <img
                        className={classes.moreCardImage}
                        src={data.image_url}
                        alt=""
                      />
                    </div>
                    <div style={{ fontWeight: "500", fontSize: "1.3rem" }}>
                      {data.name}
                    </div>
                    <div className={classes.tagContainer}>
                      <div className={classes.ingText}>겹치는 재료</div>
                      <div className={classes.tagBox}>
                        {data.shared_ingredients
                          .slice(0, 3)
                          .map((ing, index) => {
                            return <div key={index}>{ing}</div>;
                          })}
                        {/* {data.shared_ingredients.length > 3 && <div>...</div>} */}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
        {overall.length > 0 ? (
          <div className={classes.moreBox}>
            <div className={classes.moreTitle}>비슷한 음식</div>
            <div className={classes.moreItemBox}>
              {overall.map((data, idx) => {
                console.log("dd: ", data);
                return (
                  <div
                    key={idx}
                    className={classes.moreItem}
                    onClick={() => {
                      navigate(
                        `http://localhost:3000/recipe/${data.recipe_id}`,
                        {
                          state: {
                            data: { data: data },
                          },
                        }
                      );
                    }}
                  >
                    <div className={classes.moreCardImageContainer}>
                      <img
                        className={classes.moreCardImage}
                        src={data.image_url}
                        alt=""
                      />
                    </div>
                    <div style={{ fontWeight: "500", fontSize: "1.3rem" }}>
                      {data.name}
                    </div>
                    <div className={classes.tagContainer}>
                      <div className={classes.tagText}>겹치는 태그</div>
                      <div className={classes.tagBox}>
                        {data.shared_tags.slice(0, 3).map((tag, index) => {
                          return <div key={index}>{tag}</div>;
                        })}
                        {/* {data.shared_tags.length > 3 && <div>...</div>} */}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
      </div>
      <div className={classes.reviewContainer}>
        <img className={classes.reviewImage} src={reviewImage} alt="" />
      </div>
    </div>
  );
};

export default Recipe;
