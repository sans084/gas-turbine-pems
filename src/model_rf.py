import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from common import FEATURES, TARGETS, load_data, split_data, score

df = load_data()
train, test = split_data(df)
print("Train rows:", len(train), "| Test rows:", len(test))

rows = []
for target in TARGETS:
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train[target])
    pred = model.predict(test[FEATURES])
    result = score(test[target], pred)
    rows.append({"model": "Random Forest", "target": target, **result})
    print(f"{target}: MAE={result['MAE']:.3f}  RMSE={result['RMSE']:.3f}  R2={result['R2']:.3f}")

    # Which sensors did the forest rely on most?
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("  Top 3 sensors:", ", ".join(f"{k} ({v:.2f})" for k, v in imp.head(3).items()))

pd.DataFrame(rows).round(3).to_csv("reports/results_random_forest.csv", index=False)