# Progress log
## Stage 0 - Setup
Created repo, added dataset and project plan. Next: EDA.

## Stage 1 - Data loading, cleaning and EDA
- Loaded 36,733 rows (5 yearly files). Removed 7 duplicates -> 36,726 rows. No missing values.
- Distributions: NOx is roughly bell-shaped (peak ~60-65). CO is heavily skewed with a long tail up to 44 (only 500 values above 10, mean 2.37), which will make it harder to predict.
- Heatmap: NOx correlates most with ambient temperature (about -0.56); CO correlates most with turbine inlet temperature (-0.71). GTEP, TIT, TEY and CDP are highly correlated with each other. NOx and CO are only weakly correlated, so I model them separately.
- Yearly means: NOx dropped from ~70 (2013) to ~60 (2014-15), a drift between years.
- Plots saved in reports/. Next: Stage 2, Linear Regression baseline.

## Stage 2 - Linear Regression baseline
- Shared helper (src/common.py): data loading, 80/20 random split (seed 42), MAE/RMSE/R2.
- Linear Regression: NOx MAE 5.82, RMSE 8.07, R2 0.51; CO MAE 0.84, RMSE 1.47, R2 0.58.
- Only about half the variation is explained, so the relationships are probably non-linear. Next: Random Forest.

## Stage 3 - Random Forest
- Random Forest (100 trees): NOx MAE 2.61, RMSE 4.13, R2 0.87; CO MAE 0.48, RMSE 1.05, R2 0.79.
- Error roughly halved compared with Linear Regression, which confirms the relationships are non-linear.
- Feature importance agrees with the EDA: NOx depends mainly on ambient temperature (AT), CO on turbine inlet temperature (TIT).
- Next: XGBoost.

## Stage 4 - XGBoost
- XGBoost (300 trees, depth 6): NOx MAE 3.03, R2 0.85; CO MAE 0.52, R2 0.80.
- Similar to Random Forest: slightly worse on NOx, slightly better on CO R2 and RMSE. Both far better than Linear Regression.
- Both rank AT (NOx) and TIT (CO) as the most important sensors.
- Not tuned yet, so the paper's finding that XGBoost is best was not clearly reproduced. Next: MLP.