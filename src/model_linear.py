import pandas as pd
from sklearn.linear_model import LinearRegression
from common import FEATURES, TARGETS, load_data, split_data, score

df = load_data()
train, test = split_data(df)
print("Train rows:", len(train), "| Test rows:", len(test))

rows = []
for target in TARGETS:
    model = LinearRegression()
    model.fit(train[FEATURES], train[target])      # learn from the training rows
    pred = model.predict(test[FEATURES])           # predict the unseen test rows
    result = score(test[target], pred)
    rows.append({"model": "Linear Regression", "target": target, **result})
    print(f"{target}: MAE={result['MAE']:.3f}  RMSE={result['RMSE']:.3f}  R2={result['R2']:.3f}")

pd.DataFrame(rows).round(3).to_csv("reports/results_linear_regression.csv", index=False)