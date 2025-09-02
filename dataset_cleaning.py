# HIT140 Assessment 2 — Descriptive Analysis on cleaned datasets

import os
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load the cleaned datasets
print("Step 1: Loading cleaned datasets...")
d1 = pd.read_csv(
    "dataset_cleaned/dataset1_cleaned.csv",
    parse_dates=["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
)
d2 = pd.read_csv(
    "dataset_cleaned/dataset2_cleaned.csv",
    parse_dates=["time"]
)
print("Loaded cleaned datasets.\n")

# 2) Quick shapes to confirm we loaded what we expect
print("Step 2: Basic shapes")
print(f"dataset1_cleaned shape: {d1.shape}")
print(f"dataset2_cleaned shape: {d2.shape}\n")

# 3) Ensure categorical types for key columns in dataset1
print("Step 3: Ensure categorical types for key columns")
for col in ["risk", "reward", "season"]:
    if col in d1.columns and d1[col].dtype != "category":
        d1[col] = d1[col].astype("category")
print("Categorical types set where applicable.\n")

# 4) Summary tables for dataset1 (behavioural variables)
print("Step 4: Summary tables (dataset1)")
print("Value counts: risk")
print(d1["risk"].value_counts(dropna=False), "\n")

print("Value counts: reward")
print(d1["reward"].value_counts(dropna=False), "\n")

print("Crosstab: risk vs reward")
print(pd.crosstab(d1["risk"], d1["reward"]), "\n")

print("Descriptive stats: bat_landing_to_food (seconds)")
print(d1["bat_landing_to_food"].describe(), "\n")

# 5) Summary tables for dataset2 (30-min windows)
print("Step 5: Summary tables (dataset2)")
print("Descriptive stats: bat_landing_number")
print(d2["bat_landing_number"].describe(), "\n")

print("Descriptive stats: rat_arrival_number")
print(d2["rat_arrival_number"].describe(), "\n")

print("Descriptive stats: rat_minutes")
print(d2["rat_minutes"].describe(), "\n")

print("Descriptive stats: food_availability")
print(d2["food_availability"].describe(), "\n")

# 6) Prepare plots directory
print("Step 6: Create plots directory if missing")
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_DIR}\n")

# 7) Plots for dataset1
print("Step 7: Plots for dataset1")

# 7.1 Histogram: time from landing to approaching food
plt.figure()
d1["bat_landing_to_food"].plot(kind="hist", bins=40)
plt.title("Distribution of time from landing to food (dataset1)")
plt.xlabel("Seconds from landing to approaching food")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "d1_bat_landing_to_food_hist.png"))
plt.close()

# 7.2 Bar plot: Risk vs Reward counts
ct = pd.crosstab(d1["risk"], d1["reward"])
plt.figure()
ct.plot(kind="bar")
plt.title("Risk vs Reward counts (dataset1)")
plt.xlabel("Risk (0=avoid, 1=take)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "d1_risk_vs_reward_bar.png"))
plt.close()

# 7.3 Boxplot: seconds after rat arrival grouped by risk
plt.figure()
d1.boxplot(column="seconds_after_rat_arrival", by="risk")
plt.title("Seconds after rat arrival by risk (dataset1)")
plt.suptitle("")  # remove automatic super title
plt.xlabel("Risk (0=avoid, 1=take)")
plt.ylabel("Seconds after rat arrival")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "d1_seconds_after_rat_arrival_by_risk_box.png"))
plt.close()

print("Saved dataset1 plots.\n")

# 8) Plots for dataset2
print("Step 8: Plots for dataset2")

# 8.1 Scatter: rat_arrival_number vs bat_landing_number
plt.figure()
plt.scatter(d2["rat_arrival_number"], d2["bat_landing_number"])
plt.title("Rat arrivals vs Bat landings (dataset2)")
plt.xlabel("Rat arrival number (per 30 min)")
plt.ylabel("Bat landing number (per 30 min)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "d2_rat_arrivals_vs_bat_landings_scatter.png"))
plt.close()

# 8.2 Scatter: rat_minutes vs bat_landing_number
plt.figure()
plt.scatter(d2["rat_minutes"], d2["bat_landing_number"])
plt.title("Rat minutes vs Bat landings (dataset2)")
plt.xlabel("Rat minutes (per 30 min)")
plt.ylabel("Bat landing number (per 30 min)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "d2_rat_minutes_vs_bat_landings_scatter.png"))
plt.close()

# 8.3 Scatter: rat_minutes vs food_availability
plt.figure()
plt.scatter(d2["rat_minutes"], d2["food_availability"])
plt.title("Rat minutes vs Food availability (dataset2)")
plt.xlabel("Rat minutes (per 30 min)")
plt.ylabel("Food availability")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "d2_rat_minutes_vs_food_scatter.png"))
plt.close()

print("Saved dataset2 plots.\n")

# 9) Simple grouped summaries to support the narrative
print("Step 9: Simple grouped summaries")

# 9.1 Mean landing-to-food time by risk
print("Mean landing-to-food time by risk (seconds):")
print(d1.groupby("risk")["bat_landing_to_food"].mean(), "\n")

# 9.2 Mean bat landings by rat-arrival buckets
print("Mean bat landings by rat arrival buckets:")
arrivals_bucket = pd.cut(
    d2["rat_arrival_number"],
    bins=[-1, 0, 1, 3, d2["rat_arrival_number"].max()],
    labels=["0", "1", "2-3", "4+"]
)
print(d2.groupby(arrivals_bucket)["bat_landing_number"].mean(), "\n")

# 9.3 Mean food availability by rat-minutes buckets
print("Mean food availability by rat minutes buckets:")
minutes_bucket = pd.cut(
    d2["rat_minutes"],
    bins=[-0.1, 0, 5, 30, d2["rat_minutes"].max()],
    labels=["0", "0-5", "5-30", "30+"]
)
print(d2.groupby(minutes_bucket)["food_availability"].mean(), "\n")

print("Descriptive analysis complete.")