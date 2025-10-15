# seasonal_plots.py
"""
Seasonal Plots - Investigation B
HIT140 Foundations of Data Science | Group Project (Assessment 3)
Author: Suyog Kadariya
Description:
    Generates and saves visualizations comparing bat and rat behaviour
    across winter and spring using cleaned datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path Setup
os.makedirs("figures", exist_ok=True)

d1 = pd.read_csv('../dataset_cleaned/dataset1_cleaned.csv')
d2 = pd.read_csv('../dataset_cleaned/dataset2_cleaned.csv')

# Ensure consistent mapping
d1['season'] = d1['season'].map({0: 'winter', 1: 'spring'})
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

# Plot 1: Risk-taking by Season (Bar)

plt.figure(figsize=(7,5))
risk_counts = d1.groupby(['season', 'risk']).size().reset_index(name='count')
risk_pivot = risk_counts.pivot(index='season', columns='risk', values='count').fillna(0)
risk_pivot.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'])
plt.title('Risk-taking Behaviour by Season')
plt.xlabel('Season')
plt.ylabel('Count of Bats')
plt.legend(['Risk-avoidance (0)', 'Risk-taking (1)'])
plt.tight_layout()
plt.savefig('figures/risk_by_season_bar.png', dpi=300)
plt.close()
print("Saved: figures/risk_by_season_bar.png")

# Plot 2: Landing-to-Food Time Comparison (Boxplot)

plt.figure(figsize=(7,5))
sns.boxplot(data=d1, x='season', y='bat_landing_to_food', palette='coolwarm')
plt.title('Landing-to-Food Time by Season')
plt.xlabel('Season')
plt.ylabel('Time (seconds)')
plt.tight_layout()
plt.savefig('figures/landing_to_food_boxplot.png', dpi=300)
plt.close()
print("Saved: figures/landing_to_food_boxplot.png")

# Plot 3: Rat Arrivals vs Bat Landings (Scatter)

winter_data = d2[d2['season'] == 'winter']

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=winter_data,
    x='rat_arrival_number', y='bat_landing_number',
    color='#1f77b4', alpha=0.7
)
sns.regplot(
    data=winter_data,
    x='rat_arrival_number', y='bat_landing_number',
    scatter=False, color='red', line_kws={'linewidth':1.5}
)
plt.title('Winter: Rat Arrivals vs Bat Landings')
plt.xlabel('Rat Arrivals')
plt.ylabel('Bat Landings')
plt.tight_layout()
plt.savefig('figures/rat_vs_bat_winter_scatter.png', dpi=300)
plt.close()
print("Saved: figures/rat_vs_bat_winter_scatter.png")

# Plot 4: Correlation Heatmap for Dataset2

plt.figure(figsize=(7,6))
corr = d2[['bat_landing_number', 'rat_arrival_number', 'rat_minutes', 'food_availability']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Dataset2)')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap_dataset2.png', dpi=300)
plt.close()
print("Saved: figures/correlation_heatmap_dataset2.png")

print("\nAll seasonal plots successfully generated in /figures/")