// src/components/ResetButton.jsx
import React from "react";
import classes from "./ResetButton.module.css";
import { FiRefreshCw } from "react-icons/fi";
import { useSession } from "../../Context/SessionContext";

function ResetButton() {
  const { resetSession } = useSession();

  const handleReset = () => {
    // localStorage 비우기 (필요한 key만 제거)
    window.localStorage.removeItem("sessionState");

    // 세션 context 초기화
    resetSession();
  };

  return (
    <div className={classes.button} onClick={handleReset}>
      <FiRefreshCw className={classes.icon} />
      <span className={classes.label}>초기화</span>
    </div>
  );
}

export default ResetButton;
