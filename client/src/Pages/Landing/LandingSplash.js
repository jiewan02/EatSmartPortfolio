// src/components/LandingSplash.js
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import classes from "./LandingSplash.module.css";

// TODO: 실제 이미지 경로로 교체하세요.
import heroImage from "../../Assets/Images/밥상2.png";

const LandingSplash = () => {
  const [isFading, setIsFading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // 2초 안에 자연스럽게 사라지도록: 먼저 페이드 아웃, 그 다음 페이지 이동
    const fadeTimer = setTimeout(() => {
      setIsFading(true);
      // setIsFading(false);
    }, 1600); // 0.4초 정도 페이드 아웃 시간 확보

    const navTimer = setTimeout(() => {
      navigate("/home"); // 다음 페이지 경로에 맞게 수정
      // navigate("/"); // 다음 페이지 경로에 맞게 수정
    }, 2000);

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(navTimer);
    };
  }, [navigate]);

  const containerClass = `${classes.container} ${
    isFading ? classes.fadeOut : ""
  }`;

  return (
    <div className={containerClass}>
      <div className={classes.phoneFrame}>
        {/* 상단 상태바 */}
        <div className={classes.statusBar}>
          <div className={classes.statusLeft}>8:30</div>
          <div className={classes.statusCenter} />
          <div className={classes.statusRight}>
            <span className={classes.icon}>5G</span>
            <span className={classes.battery} />
          </div>
        </div>

        {/* 메인 이미지 */}
        <img
          src={heroImage}
          alt="오늘은 뭐 먹지? 음식 이미지"
          className={classes.heroImage}
        />

        {/* 텍스트 오버레이 */}
        <div className={classes.textOverlay}>
          <p className={classes.question}>해 먹고 말지!</p>
          {/* 오렌지 박스 */}
          <div className={classes.dingdongBox}>
            <span className={classes.dingdongText}>AI 레시피 검색</span>
            {/* 흰색 shadow bar */}
            <div className={classes.bottomShadow} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingSplash;
