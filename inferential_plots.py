'''

HIT140 Assessment 2 — Visualising inferential results (for our use and more clear understanding in graph)

Group Name: SYDN 28
Group Members:
Krish Rajbhandari - S395754
Tasnim Zannat - S394294
Asma Zia - S395083
Suyog Kadariya - S393829

This script makes clear pictures of our inferential results.
We plot the chi-square table, the group difference boxplot, the logistic regression ORs,
and the two correlations so the stats are easy to see and explain in the presentation.

'''


# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

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

# Ensure categorical where appropriate
for col in ["risk", "reward", "season"]:
    if col in d1.columns and d1[col].dtype.name != "category":
        d1[col] = d1[col].astype("category")

print("Step 2: Preparing output folder for plots...")
PLOT_DIR = "inferential_plots"
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOT_DIR}\n")

# -----------------------------
# Chi-square: risk x reward
# -----------------------------
print("Step 3: Chi-square visualisations...")
ct = pd.crosstab(d1["risk"], d1["reward"])  # counts

# 3a) Counts bar chart
plt.figure()
ct.plot(kind="bar")
plt.title("Risk × Reward (counts)")
plt.xlabel("Risk (0=avoid, 1=take)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "chi_counts_risk_reward.png"))
plt.close()

# 3b) % stacked bar chart
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
plt.figure()
ct_pct.plot(kind="bar", stacked=True)
plt.title("Risk × Reward (% within risk group)")
plt.xlabel("Risk (0=avoid, 1=take)")
plt.ylabel("Percent")
plt.legend(title="Reward (0/1)", loc="best")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "chi_percent_stacked_risk_reward.png"))
plt.close()

# Compute statistics for a caption (optional to print)
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
n = ct.values.sum()
phi2 = chi2 / n
r, k = ct.shape
cramers_v = np.sqrt(phi2 / (min(k - 1, r - 1)))
print(f"Chi-square done: chi2={chi2:.3f}, df={dof}, p={p_chi:.3g}, Cramer's V={cramers_v:.3f}")

# ---------------------------------------------
# Mann–Whitney U: landing -> food time by risk
# ---------------------------------------------
print("Step 4: Mann–Whitney visualisation...")
x = d1.loc[d1["risk"] == 0, "bat_landing_to_food"]
y = d1.loc[d1["risk"] == 1, "bat_landing_to_food"]
u_stat, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
rank_biserial = 1 - (2 * u_stat) / (len(x) * len(y))

plt.figure()
data = [x, y]
plt.boxplot(data, tick_labels=["risk=0", "risk=1"], showfliers=False)
plt.title("Landing→Food time by risk (boxplot, outliers hidden)")
plt.xlabel("Risk group")
plt.ylabel("Seconds from landing to approaching food")
# annotate p-value and effect size
txt = f"Mann–Whitney U p={p_u:.2e}, rank-biserial={rank_biserial:.3f}\nMeans: 0→{x.mean():.2f}s, 1→{y.mean():.2f}s"
plt.gcf().text(0.5, -0.12, txt, ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "mannwhitney_boxplot_landing_to_food_by_risk.png"), bbox_inches="tight")
plt.close()

# -------------------------------------------------------
# Logistic regression: ORs with 95% CI (per 1 minute)
# -------------------------------------------------------
print("Step 5: Logistic regression visualisations...")
# Rescale seconds to minutes for interpretability
d1_model = d1.copy()
d1_model["risk_num"] = d1_model["risk"].cat.codes
d1_model["sec_minutes"] = d1_model["seconds_after_rat_arrival"] / 60.0

model = smf.logit("risk_num ~ sec_minutes + C(season)", data=d1_model).fit(disp=False)
params = model.params
conf = model.conf_int()
or_df = pd.DataFrame({
    "term": params.index,
    "OR": np.exp(params.values),
    "CI_lower": np.exp(conf[0].values),
    "CI_upper": np.exp(conf[1].values),
    "p_value": model.pvalues.values
})

# Drop Intercept for the forest plot
or_plot = or_df[or_df["term"] != "Intercept"].copy()

# Forest plot of ORs with 95% CI
plt.figure()
ypos = np.arange(len(or_plot))
plt.errorbar(or_plot["OR"], ypos, xerr=[or_plot["OR"] - or_plot["CI_lower"], or_plot["CI_upper"] - or_plot["OR"]],
             fmt='o', capsize=4)
plt.axvline(1.0, linestyle="--")
plt.yticks(ypos, or_plot["term"])
plt.xlabel("Odds ratio (logit)")
plt.title("Logistic regression ORs with 95% CI\n(Outcome: risk=1; Predictor per +1 minute since rat arrival)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "logit_forest_odds_ratios.png"))
plt.close()

# Predicted probability vs sec_minutes by season (optional)
grid = pd.DataFrame({
    "sec_minutes": np.linspace(d1_model["sec_minutes"].min(), d1_model["sec_minutes"].max(), 100),
    "season": 0
})
grid1 = grid.copy()
grid1["season"] = 1
pred0 = model.predict(grid)     # season=0
pred1 = model.predict(grid1)    # season=1

plt.figure()
plt.plot(grid["sec_minutes"], pred0, label="season=0")
plt.plot(grid1["sec_minutes"], pred1, label="season=1")
plt.xlabel("Minutes since rat arrival")
plt.ylabel("Predicted probability of risk=1")
plt.title("Predicted risk-taking vs time since rat arrival by season")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "logit_pred_prob_vs_minutes_by_season.png"))
plt.close()

# -------------------------------------------------------
# Correlations: scatter + simple trend lines
# -------------------------------------------------------
print("Step 6: Correlation visualisations...")

# 6a) Rat arrivals vs bat landings
x1 = d2["rat_arrival_number"].values
y1 = d2["bat_landing_number"].values
m1, b1 = np.polyfit(x1, y1, 1)
rho1, p1 = stats.spearmanr(x1, y1)
plt.figure()
plt.scatter(x1, y1, alpha=0.35)
xx = np.linspace(x1.min(), x1.max(), 100)
plt.plot(xx, m1 * xx + b1)
plt.title(f"Rat arrivals vs Bat landings (Spearman ρ={rho1:.3f}, p={p1:.2e})")
plt.xlabel("Rat arrival number (per 30 min)")
plt.ylabel("Bat landing number (per 30 min)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "corr_rat_arrivals_vs_bat_landings.png"))
plt.close()

# 6b) Rat minutes vs food availability
x2 = d2["rat_minutes"].values
y2 = d2["food_availability"].values
m2, b2 = np.polyfit(x2, y2, 1)
rho2, p2 = stats.spearmanr(x2, y2)
plt.figure()
plt.scatter(x2, y2, alpha=0.35)
xx2 = np.linspace(x2.min(), x2.max(), 100)
plt.plot(xx2, m2 * xx2 + b2)
plt.title(f"Rat minutes vs Food availability (Spearman ρ={rho2:.3f}, p={p2:.2e})")
plt.xlabel("Rat minutes (per 30 min)")
plt.ylabel("Food availability")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "corr_rat_minutes_vs_food.png"))
plt.close()

print("All inferential plots saved.")