import { useNavigate, useParams } from "react-router-dom";
import classes from "./NavBar.module.css";
import { useSession } from "../../Context/SessionContext";
const NavBar = () => {
  const navigate = useNavigate();
  const { recommendations } = useSession();
  const { id } = useParams();
  return (
    <div className={classes.headerContainer}>
      <div className={classes.statusBar}>
        <div className={classes.statusLeft}>8:30</div>
        <div className={classes.statusCenter} />
        <div className={classes.statusRight}>
          <span className={classes.icon}>5G</span>
          <span className={classes.battery} />
        </div>
      </div>
      <div className={classes.hamburgerContainer}>
        <div className={classes.hamburger}>
          <span className={classes.line}></span>
          <span className={classes.line}></span>
          <span className={classes.line}></span>
        </div>
      </div>
      <div
        className={classes.headerText}
        onClick={() => {
          navigate("/");
        }}
      >
        {recommendations.length > 0 ? (
          id ? (
            <>
              <div>주문하신</div>
              <span>메뉴 나왔습니다.</span>
            </>
          ) : (
            "레시피 검색 결과"
          )
        ) : (
          "해 먹고 말지!"
        )}
      </div>
    </div>
  );
};

export default NavBar;
