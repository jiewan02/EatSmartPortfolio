// Spinner.js
import classes from "./Spinner.module.css";

export default function Spinner({ size = "small" }) {
  return (
    <div className={classes.spinnerContainer}>
      <div className={`${classes.spinner} ${classes[size]}`}>
        <div className={classes.dot}></div>
        <div className={classes.dot}></div>
        <div className={classes.dot}></div>
        <div className={classes.dot}></div>
        <div className={classes.dot}></div>
        <div className={classes.dot}></div>
        <div className={classes.dot}></div>
        <div className={classes.dot}></div>
      </div>
    </div>
  );
}
