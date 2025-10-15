# seasonal_analysis.py
"""
Seasonal Analysis - Investigation B
HIT140 Foundations of Data Science | Group Project (Assessment 3)
Author: Suyog Kadariya
Description:
    This script performs statistical analysis comparing bat and rat behaviours
    across different seasons (winter vs spring) using cleaned datasets.
"""

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
import statsmodels.formula.api as smf

# -----------------------------
# 1. Load Cleaned Datasets
# -----------------------------
d1 = pd.read_csv('../dataset_cleaned/dataset1_cleaned.csv')
d2 = pd.read_csv('../dataset_cleaned/dataset2_cleaned.csv')

print("✅ Datasets loaded successfully!")
print(f"Dataset1 shape: {d1.shape}")
print(f"Dataset2 shape: {d2.shape}")

# -----------------------------
# 2. Add season info to dataset2 (based on month)
# -----------------------------
month_to_season = {
    'June': 'winter', 'July': 'winter', 'August': 'winter',
    'September': 'spring', 'October': 'spring', 'November': 'spring'
}

if 'season' not in d2.columns:
    d2['month'] = d2['month'].astype(str).str.strip().str.title()
    d2['season'] = d2['month'].map(month_to_season)

# -----------------------------
# 3. Chi-Square: Risk vs Season
# -----------------------------
print("\n=== Chi-Square: Risk vs Season ===")
cont_risk = pd.crosstab(d1['season'], d1['risk'])
chi2_risk, p_risk, _, _ = chi2_contingency(cont_risk)
print(f"Chi-square = {chi2_risk:.3f}, p = {p_risk:.4f}")

# -----------------------------
# 4. Chi-Square: Reward vs Season
# -----------------------------
print("\n=== Chi-Square: Reward vs Season ===")
cont_reward = pd.crosstab(d1['season'], d1['reward'])
chi2_reward, p_reward, _, _ = chi2_contingency(cont_reward)
print(f"Chi-square = {chi2_reward:.3f}, p = {p_reward:.4f}")

# -----------------------------
# 5. Mann–Whitney U: Landing Speed (Winter vs Spring)
# -----------------------------
print("\n=== Mann–Whitney U: Landing Time ===")

# Ensure 'season' is treated as string
d1['season'] = d1['season'].astype(str).str.lower().str.strip()

# Filter safely
winter = d1[d1['season'] == 'winter']
spring = d1[d1['season'] == 'spring']

# Ensure numeric landing times and handle small samples
winter_times = pd.to_numeric(winter['bat_landing_to_food'], errors='coerce').dropna()
spring_times = pd.to_numeric(spring['bat_landing_to_food'], errors='coerce').dropna()

if len(winter_times) > 10 and len(spring_times) > 10:
    u_stat, p_val = mannwhitneyu(winter_times, spring_times, alternative='two-sided')
    print(f"Mann–Whitney U = {u_stat:.3f}, p = {p_val:.4f}")
else:
    print(f"⚠️ Not enough valid samples for Mann–Whitney test. Winter={len(winter_times)}, Spring={len(spring_times)}")
    u_stat, p_val = float('nan'), float('nan')

# -----------------------------
# 6. Spearman Correlation: Rat arrivals vs Bat landings per season
# -----------------------------
print("\n=== Spearman Correlations (Dataset2) ===")
results_corr = []

# Ensure consistency
d2['season'] = d2['season'].astype(str).str.lower().str.strip()

for season in ['winter', 'spring']:
    sub = d2[d2['season'] == season]
    if not sub.empty:
        corr, p_corr = spearmanr(sub['rat_arrival_number'], sub['bat_landing_number'])
        results_corr.append((season, corr, p_corr))
        print(f"{season.capitalize()} → Spearman ρ = {corr:.3f}, p = {p_corr:.4f}")
    else:
        print(f"⚠️ No data found for {season} season in dataset2.")
        results_corr.append((season, float('nan'), float('nan')))

# -----------------------------
# 7. Logistic Regression (Optional, adds depth)
# -----------------------------
print("\n=== Logistic Regression: Risk ~ Season + Rat Presence + Hours After Sunset ===")
try:
    model = smf.logit('risk ~ C(season) + seconds_after_rat_arrival + hours_after_sunset', data=d1).fit()
    print(model.summary())
except Exception as e:
    print("⚠️ Logistic regression could not be computed:", e)

# -----------------------------
# 8. Save Results to Summary File
# -----------------------------
with open('seasonal_results_summary.txt', 'w') as f:
    f.write("=== Seasonal Analysis Results ===\n\n")
    f.write(f"Chi-square (Risk vs Season): {chi2_risk:.3f}, p={p_risk:.4f}\n")
    f.write(f"Chi-square (Reward vs Season): {chi2_reward:.3f}, p={p_reward:.4f}\n")
    f.write(f"Mann–Whitney U (Landing Speed): U={u_stat:.3f}, p={p_val:.4f}\n\n")
    for s, c, p in results_corr:
        f.write(f"Spearman ({s.title()}): rho={c:.3f}, p={p:.4f}\n")
    f.write("\nLogistic regression summary is printed in console.\n")

print("\n✅ Results saved to seasonal_results_summary.txt")
