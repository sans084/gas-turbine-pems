import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from common import FEATURES, TARGETS, load_data, split_data, score

df = load_data()
train, test = split_data(df)
print("Train rows:", len(train), "| Test rows:", len(test))

rows = []
for target in TARGETS:
    # Scale the inputs, then a small neural network (two hidden layers: 64 and 32 neurons).
    # The target is scaled too, because neural networks learn better with small numbers.
    net = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                     early_stopping=True, random_state=42),
    )
    model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
    model.fit(train[FEATURES], train[target])
    pred = model.predict(test[FEATURES])
    result = score(test[target], pred)
    rows.append({"model": "MLP", "target": target, **result})
    print(f"{target}: MAE={result['MAE']:.3f}  RMSE={result['RMSE']:.3f}  R2={result['R2']:.3f}")

pd.DataFrame(rows).round(3).to_csv("reports/results_mlp.csv", index=False)