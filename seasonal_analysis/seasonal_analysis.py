# seasonal_analysis.py
"""
Seasonal Analysis - Investigation B
HIT140 Foundations of Data Science | Group Project (Assessment 3)
Author: Suyog Kadariya
Description:
    Statistical comparison of bat and rat behaviours across winter and spring
    using cleaned datasets. This version uses verified numeric-to-season mapping
    based on the official Project and Data Brief.
"""

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
import statsmodels.formula.api as smf

# Load Cleaned Datasets
d1 = pd.read_csv('../dataset_cleaned/dataset1_cleaned.csv')
d2 = pd.read_csv('../dataset_cleaned/dataset2_cleaned.csv')

print("Datasets loaded successfully!")
print(f"Dataset1 shape: {d1.shape}")
print(f"Dataset2 shape: {d2.shape}")

# Map numeric seasons in dataset1 to actual names
# (0 = winter, 1 = spring)
if 'season' in d1.columns:
    d1['season'] = d1['season'].map({0: 'winter', 1: 'spring'})
else:
    print("No 'season' column found in dataset1 — please verify column names.")

# Convert numeric months in dataset2 and assign seasons
# (Winter = June–August & Spring = September–November)
def month_to_season(m):
    try:
        if int(m) in [5, 6, 7]:
            return 'winter'
        elif int(m) in [8, 9, 10]:
            return 'spring'
        else:
            return None
    except:
        return None

d2['season'] = d2['month'].apply(month_to_season)

# Chi-Square: Risk vs Season
print("\n=== Chi-Square: Risk vs Season ===")
cont_risk = pd.crosstab(d1['season'], d1['risk'])
chi2_risk, p_risk, _, _ = chi2_contingency(cont_risk)
print(f"Chi-square = {chi2_risk:.3f}, p = {p_risk:.4f}")

# Chi-Square: Reward vs Season
print("\n=== Chi-Square: Reward vs Season ===")
cont_reward = pd.crosstab(d1['season'], d1['reward'])
chi2_reward, p_reward, _, _ = chi2_contingency(cont_reward)
print(f"Chi-square = {chi2_reward:.3f}, p = {p_reward:.4f}")

# Mann–Whitney U: Landing Speed (Winter vs Spring)
print("\n=== Mann–Whitney U: Landing Time ===")
winter = d1[d1['season'] == 'winter']
spring = d1[d1['season'] == 'spring']

winter_times = pd.to_numeric(winter['bat_landing_to_food'], errors='coerce').dropna()
spring_times = pd.to_numeric(spring['bat_landing_to_food'], errors='coerce').dropna()

if len(winter_times) > 10 and len(spring_times) > 10:
    u_stat, p_val = mannwhitneyu(winter_times, spring_times, alternative='two-sided')
    print(f"Mann–Whitney U = {u_stat:.3f}, p = {p_val:.4f}")
else:
    print(f"Not enough valid samples for Mann–Whitney test. Winter={len(winter_times)}, Spring={len(spring_times)}")
    u_stat, p_val = float('nan'), float('nan')

# Spearman Correlation: Rat arrivals vs Bat landings per season
print("\n=== Spearman Correlations (Dataset2) ===")
results_corr = []

for season in ['winter', 'spring']:
    sub = d2[d2['season'] == season]
    if not sub.empty:
        corr, p_corr = spearmanr(sub['rat_arrival_number'], sub['bat_landing_number'])
        results_corr.append((season, corr, p_corr))
        print(f"{season.capitalize()} → Spearman ρ = {corr:.3f}, p = {p_corr:.4f}")
    else:
        print(f"No data found for {season} season in dataset2.")
        results_corr.append((season, float('nan'), float('nan')))

# Logistic Regression to add analytical depth
print("\n=== Logistic Regression: Risk ~ Season + Rat Presence + Hours After Sunset ===")
try:
    model = smf.logit('risk ~ C(season) + seconds_after_rat_arrival + hours_after_sunset', data=d1).fit()
    print(model.summary())
except Exception as e:
    print("Logistic regression could not be computed:", e)

# Save Results to Summary File
with open('seasonal_results_summary.txt', 'w') as f:
    f.write("=== Seasonal Analysis Results ===\n\n")
    f.write(f"Chi-square (Risk vs Season): {chi2_risk:.3f}, p={p_risk:.4f}\n")
    f.write(f"Chi-square (Reward vs Season): {chi2_reward:.3f}, p={p_reward:.4f}\n")
    f.write(f"Mann–Whitney U (Landing Speed): U={u_stat:.3f}, p={p_val:.4f}\n\n")
    for s, c, p in results_corr:
        f.write(f"Spearman ({s.title()}): rho={c:.3f}, p={p:.4f}\n")
    f.write("\nLogistic regression summary printed in console.\n")

print("\nResults saved to seasonal_results_summary.txt")
