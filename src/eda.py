import glob
import pandas as pd

# Read the five yearly files and stack them into one table
frames = []
for f in sorted(glob.glob("data/gt_*.csv")):
    d = pd.read_csv(f)
    d["year"] = int(f[-8:-4])      # take the year from the file name
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

print("Rows loaded:", len(df))
print(df.head())
print("Missing values per column:")
print(df.isna().sum())

# ---- Part 2: cleaning ----
rows_before = len(df)
df = df.drop_duplicates().dropna()
print("\nRows before cleaning:", rows_before)
print("Rows after cleaning: ", len(df))
print("Rows removed:        ", rows_before - len(df))

# Summary statistics of every column
print(df.describe().round(2))


# ---- Part 3: plots ----
import os
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

# 1. How are the two targets distributed?
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df["NOX"].hist(bins=60, ax=ax[0]); ax[0].set_title("NOx distribution (mg/m3)")
df["CO"].hist(bins=60, ax=ax[1]);  ax[1].set_title("CO distribution (mg/m3)")
plt.tight_layout()
plt.savefig("reports/01_target_distributions.png", dpi=130)
plt.close()
print("CO values above 10:", (df["CO"] > 10).sum(), "of", len(df))

# 2. Which sensors are related to the targets?
corr = df.drop(columns="year").corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.colorbar()
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("reports/02_correlation.png", dpi=130)
plt.close()

# 3. Did emissions change over the years?
yearly = df.groupby("year")[["NOX", "CO"]].mean().round(2)
print(yearly)
yearly.plot(kind="bar", figsize=(7, 4), title="Mean emissions per year")
plt.tight_layout()
plt.savefig("reports/03_yearly_means.png", dpi=130)
plt.close()