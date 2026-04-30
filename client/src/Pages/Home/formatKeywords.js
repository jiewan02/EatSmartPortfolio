const formatKeywords = (obj) => {
  const include = [];
  const exclude = [];

  // helper: include 배열에 동일 name이 이미 있는지 체크
  const addUniqueInclude = (name, field, state = "include") => {
    if (!include.some((item) => item.name === name)) {
      include.push({ name, field, state });
    }
  };

  const addUniqueExclude = (name, field, state = "exclude") => {
    if (!exclude.some((item) => item.name === name)) {
      exclude.push({ name, field, state });
    }
  };

  obj.difficulty &&
    obj.difficulty.forEach((str) => addUniqueInclude(str, "difficulty"));

  obj.dish_type &&
    obj.dish_type.forEach((str) => addUniqueInclude(str, "dish_type"));

  obj.exclude_ingredients &&
    obj.exclude_ingredients.forEach((str) =>
      addUniqueExclude(str, "exclude_ingredients")
    );

  obj.extra_keywords &&
    obj.extra_keywords.forEach((str) => {
      console.log("health", obj.health_tags, str);
      if (!obj.positive_tags.includes(str) && !obj.health_tags.includes(str)) {
        addUniqueInclude(str, "extra_keywords");
      }
    });

  obj.health_tags &&
    obj.health_tags.forEach((str) => {
      if (
        !obj.positive_tags.includes(str) &&
        !obj.extra_keywords.includes(str)
      ) {
        addUniqueInclude(str, "health_tags");
      }
    });

  obj.weather_tags &&
    obj.weather_tags.forEach((str) => addUniqueInclude(str, "weather_tags"));
  obj.situation &&
    obj.situation.forEach((str) => addUniqueInclude(str, "situation"));

  obj.max_cook_time_min &&
    addUniqueInclude(obj.max_cook_time_min, "max_cook_time_min");

  obj.menu_style &&
    obj.menu_style.forEach((str) => addUniqueInclude(str, "menu_style"));

  obj.method && obj.method.forEach((str) => addUniqueInclude(str, "method"));

  obj.must_ingredients &&
    obj.must_ingredients.forEach((str) =>
      addUniqueInclude(str, "must_ingredients")
    );

  obj.optional_ingredients &&
    obj.optional_ingredients.forEach((str) =>
      addUniqueInclude(str, "optional_ingredients")
    );
  return { include, exclude };
};

export default formatKeywords;
