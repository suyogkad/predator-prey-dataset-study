'''

HIT140 Assessment 2: Part III — Inferential Analysis on cleaned datasets

Group Name: SYDN 28
Group Members:
Krish Rajbhandari - S395754
Tasnim Zannat - S394294
Asma Zia - S395083
Suyog Kadariya - S393829

This script runs the main statistical tests on our cleaned data.
We check if risk and reward are related, compare groups, and fit a simple model.
The goal is to back up our findings with proper tests and clear numbers.

'''


# Import libraries
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# 1) Load cleaned data
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

# Ensure categorical types for d1
for col in ["risk", "reward", "season"]:
    if col in d1.columns and d1[col].dtype.name != "category":
        d1[col] = d1[col].astype("category")

# =============== DATASET 1 TESTS (individual landings) ===============

print("Step 2: Chi-square test — association between risk and reward (dataset1)")
ct = pd.crosstab(d1["risk"], d1["reward"])
chi2, p, dof, expected = stats.chi2_contingency(ct)
n = ct.values.sum()
phi2 = chi2 / n
r, k = ct.shape
cramers_v = np.sqrt(phi2 / (min(k - 1, r - 1)))
print("Contingency table:")
print(ct)
print(f"chi2={chi2:.3f}, df={dof}, p={p:.6f}")
print(f"Cramer's V={cramers_v:.3f}\n")

print("Step 3: Mann–Whitney U — landing→food time by risk (dataset1)")
x = d1.loc[d1["risk"] == 0, "bat_landing_to_food"]
y = d1.loc[d1["risk"] == 1, "bat_landing_to_food"]
u_stat, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
rank_biserial = 1 - (2 * u_stat) / (len(x) * len(y))
print(f"Group sizes: risk=0 (n={len(x)}), risk=1 (n={len(y)})")
print(f"U={u_stat:.1f}, p={p_u:.6f}")
print(f"Rank-biserial effect size={rank_biserial:.3f}")
print(f"Means: risk=0 -> {x.mean():.3f}s, risk=1 -> {y.mean():.3f}s\n")

print("Step 4: Logistic regression — does time since rat arrival predict risk? (dataset1)")
# Model: binary risk (0/1) ~ seconds_after_rat_arrival + season
d1_model = d1.copy()
d1_model["risk_num"] = d1_model["risk"].cat.codes  # ensures 0/1 numeric
model = smf.logit("risk_num ~ seconds_after_rat_arrival + C(season)", data=d1_model).fit(disp=False)
print(model.summary())

# Odds ratios with 95% CI
params = model.params
conf = model.conf_int()
or_table = pd.DataFrame({
    "OR": np.exp(params),
    "CI_lower": np.exp(conf[0]),
    "CI_upper": np.exp(conf[1]),
    "p_value": model.pvalues
})
print("\nOdds ratios (with 95% CI):")
print(or_table, "\n")

# =============== DATASET 2 TESTS (30-min windows) ===============

print("Step 5: Correlation — rat arrivals vs bat landings (dataset2)")
rho1, p1 = stats.spearmanr(d2["rat_arrival_number"], d2["bat_landing_number"])
print(f"Spearman rho={rho1:.3f}, p={p1:.6f}\n")

print("Step 6: Correlation — rat minutes vs food availability (dataset2)")
rho2, p2 = stats.spearmanr(d2["rat_minutes"], d2["food_availability"])
print(f"Spearman rho={rho2:.3f}, p={p2:.6f}\n")

print("Inferential analysis complete.")
