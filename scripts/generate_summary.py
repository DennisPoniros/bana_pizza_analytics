"""
BANA255 Pizza Survey - Summary Statistics & Visualizations
Generates plots and tables for the executive summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
questions = df.iloc[0].to_dict()
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

print("=" * 70)
print("BANA255 PIZZA SURVEY - SUMMARY REPORT")
print("=" * 70)
print(f"Total Valid Responses: {len(data)}")
print(f"Survey Period: Nov 9-14, 2025")
print("=" * 70)

# =============================================================================
# FIGURE 1: Local vs Chain Preference
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart for overall preference
q17_counts = data['Q17'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
axes[0].pie(q17_counts.values, labels=q17_counts.index, autopct='%1.1f%%',
            colors=colors, explode=(0.05, 0, 0), startangle=90)
axes[0].set_title('Q17: Preferred Business Type\n(Local vs Chain)', fontsize=12, fontweight='bold')

# Bar chart excluding "Unsure"
local_chain = data[data['Q17'].isin(['Local', 'Chain'])]['Q17'].value_counts()
bars = axes[1].bar(local_chain.index, local_chain.values, color=['#2ecc71', '#e74c3c'])
axes[1].set_ylabel('Number of Respondents')
axes[1].set_title('Clear Preference: Local vs Chain\n(Excluding "Unsure")', fontsize=12, fontweight='bold')
for bar, val in zip(bars, local_chain.values):
    pct = val / local_chain.sum() * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}\n({pct:.1f}%)', ha='center', fontsize=11)
axes[1].set_ylim(0, max(local_chain.values) * 1.2)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_local_vs_chain.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig1_local_vs_chain.png")

# =============================================================================
# FIGURE 2: Top Pizza Places
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

q28_counts = data['Q28'].value_counts().head(10)
colors = ['#3498db' if 'Domino' in x or 'Papa' in x or 'Little' in x or 'Pizza Hut' in x
          or 'Costco' in x or 'Blaze' in x else '#2ecc71' for x in q28_counts.index]

bars = ax.barh(q28_counts.index[::-1], q28_counts.values[::-1], color=colors[::-1])
ax.set_xlabel('Number of Votes')
ax.set_title('Top 10 Favorite Pizza Places\n(Blue = Chain, Green = Local)', fontsize=12, fontweight='bold')

for bar, val in zip(bars, q28_counts.values[::-1]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_top_pizza_places.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig2_top_pizza_places.png")

# =============================================================================
# FIGURE 3: Pizza Quality Importance Ratings
# =============================================================================
importance_map = {
    'Not at all important': 1, 'Slightly important': 2,
    'Moderately important': 3, 'Very important': 4, 'Extremely important': 5
}

q5_labels = {
    'Q5_1': 'Taste & Flavor',
    'Q5_2': 'Ingredient Quality',
    'Q5_3': 'Crust Excellence',
    'Q5_4': 'Balance & Ratios',
    'Q5_5': 'Freshness & Temp',
    'Q5_6': 'Appearance',
    'Q5_7': 'Price & Value',
    'Q5_8': 'Convenience',
    'Q5_9': 'Special Features'
}

importance_data = []
for col, label in q5_labels.items():
    scores = data[col].map(importance_map).dropna()
    importance_data.append({
        'Characteristic': label,
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Pct_High': ((scores >= 4).sum() / len(scores)) * 100
    })

importance_df = pd.DataFrame(importance_data).sort_values('Mean', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))
bars = ax.barh(importance_df['Characteristic'], importance_df['Mean'],
               xerr=importance_df['Std'], color=colors, capsize=3)
ax.set_xlabel('Mean Importance Rating (1-5 scale)')
ax.set_title('What Makes Pizza "The Best"?\nImportance Ratings by Characteristic', fontsize=12, fontweight='bold')
ax.set_xlim(1, 5)
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='Neutral (3)')

for bar, pct in zip(bars, importance_df['Pct_High']):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
            f'{pct:.0f}% high', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_importance_ratings.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig3_importance_ratings.png")

# =============================================================================
# FIGURE 4: Order Method & Time Preferences
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Order method pie
q11_counts = data['Q11'].value_counts()
q11_labels = ['Pick up', 'Delivery', 'Third-party App']
axes[0].pie(q11_counts.values, labels=q11_labels, autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c', '#9b59b6'], startangle=90)
axes[0].set_title('How Do You Get Your Pizza?', fontsize=12, fontweight='bold')

# Time comparison
data['Q14_1_num'] = pd.to_numeric(data['Q14_1'], errors='coerce')
data['Q14_2_num'] = pd.to_numeric(data['Q14_2'], errors='coerce')
data['Q15_1_num'] = pd.to_numeric(data['Q15_1'], errors='coerce')
data['Q15_2_num'] = pd.to_numeric(data['Q15_2'], errors='coerce')

time_data = {
    'Category': ['Delivery\nExpected', 'Delivery\nFor Best', 'Pickup\nExpected', 'Pickup\nFor Best'],
    'Minutes': [data['Q14_1_num'].mean(), data['Q15_1_num'].mean(),
                data['Q14_2_num'].mean(), data['Q15_2_num'].mean()]
}
time_df = pd.DataFrame(time_data)
colors = ['#3498db', '#2ecc71', '#3498db', '#2ecc71']
bars = axes[1].bar(time_df['Category'], time_df['Minutes'], color=colors)
axes[1].set_ylabel('Minutes')
axes[1].set_title('Time: Expected vs Willing to Wait for "Best"\n(Blue=Expected, Green=For Best)',
                  fontsize=12, fontweight='bold')
for bar, val in zip(bars, time_df['Minutes']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_order_time_preferences.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig4_order_time_preferences.png")

# =============================================================================
# FIGURE 5: Key Decision Factors
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Price vs Location
q16_counts = data['Q16'].value_counts()
axes[0].pie(q16_counts.values, labels=q16_counts.index, autopct='%1.1f%%',
            colors=['#2ecc71', '#3498db'], explode=(0.05, 0), startangle=90)
axes[0].set_title('More Important: Price or Location?', fontsize=12, fontweight='bold')

# Factor comparison
factors = ['Taste', 'Price/Value', 'Convenience', 'Variety']
factor_means = [
    data['Q5_1'].map(importance_map).mean(),
    data['Q5_7'].map(importance_map).mean(),
    data['Q5_8'].map(importance_map).mean(),
    data['Q8'].map(importance_map).mean()
]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, 4))[::-1]
bars = axes[1].bar(factors, factor_means, color=colors)
axes[1].set_ylabel('Mean Importance (1-5)')
axes[1].set_title('Key Decision Factors Ranked', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 5)
axes[1].axhline(y=3, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, factor_means):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_decision_factors.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig5_decision_factors.png")

# =============================================================================
# FIGURE 6: Demographics Overview
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
data['Q30_num'] = pd.to_numeric(data['Q30'], errors='coerce')
axes[0, 0].hist(data['Q30_num'].dropna(), bins=range(17, 28), color='#3498db', edgecolor='white', alpha=0.8)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Age Distribution', fontsize=11, fontweight='bold')
axes[0, 0].axvline(data['Q30_num'].mean(), color='red', linestyle='--', label=f'Mean: {data["Q30_num"].mean():.1f}')
axes[0, 0].legend()

# Year in school
q31_order = ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Super Senior', 'Graduate student']
q31_counts = data['Q31'].value_counts().reindex(q31_order).dropna()
axes[0, 1].bar(range(len(q31_counts)), q31_counts.values, color='#9b59b6')
axes[0, 1].set_xticks(range(len(q31_counts)))
axes[0, 1].set_xticklabels(['Fresh', 'Soph', 'Junior', 'Senior', 'Super Sr', 'Grad'], rotation=45)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Year in School', fontsize=11, fontweight='bold')

# Gender
q32_counts = data['Q32'].value_counts()
axes[1, 0].pie(q32_counts.values, labels=q32_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette("pastel"), startangle=90)
axes[1, 0].set_title('Gender Distribution', fontsize=11, fontweight='bold')

# Residence
q33_counts = data['Q33'].value_counts()
axes[1, 1].pie(q33_counts.values, labels=q33_counts.index, autopct='%1.1f%%',
               colors=['#3498db', '#2ecc71', '#e74c3c'], startangle=90)
axes[1, 1].set_title('Residence While at RIT', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig6_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig6_demographics.png")

# =============================================================================
# SUMMARY STATISTICS TABLE
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

summary_stats = """
┌─────────────────────────────────────────────────────────────────────┐
│                    SAMPLE CHARACTERISTICS                           │
├─────────────────────────────────────────────────────────────────────┤
│  Total Responses (Consented)     │  161                             │
│  Mean Age                        │  {age:.1f} years                     │
│  Median Pizza Orders/Month       │  {orders:.0f}                               │
│  Has Transportation              │  {transport:.1f}%                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    KEY FINDINGS SUMMARY                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. LOCAL VS CHAIN                                                  │
│     • {local_pct:.1f}% prefer local (when choosing between local/chain)   │
│     • Yet Domino's (chain) is #1 favorite ({dominos_pct:.1f}% of votes)       │
│     • Statistical significance: p < 0.001                           │
├─────────────────────────────────────────────────────────────────────┤
│  2. VISIT FREQUENCY                                                 │
│     • Mean: {mean_visits:.2f} orders/month                                   │
│     • Median: {med_visits:.0f} orders/month                                    │
│     • Range: 0-15 orders/month                                      │
├─────────────────────────────────────────────────────────────────────┤
│  3. MOST IMPORTANT CHARACTERISTICS (Mean out of 5)                  │
│     • Taste & Flavor: {taste:.2f} (94% rate highly important)           │
│     • Balance & Ratios: {balance:.2f}                                       │
│     • Crust Excellence: {crust:.2f}                                        │
├─────────────────────────────────────────────────────────────────────┤
│  4. ORDER METHOD                                                    │
│     • Pickup: {pickup_pct:.1f}%                                             │
│     • Delivery: {delivery_pct:.1f}%                                           │
│     • Third-party apps: {app_pct:.1f}%                                       │
├─────────────────────────────────────────────────────────────────────┤
│  5. DECISION FACTORS (Ranked by importance)                         │
│     • 1st: Taste ({taste:.2f}/5)                                        │
│     • 2nd: Price/Value ({price:.2f}/5)                                   │
│     • 3rd: Convenience ({conv:.2f}/5)                                   │
│     • 4th: Variety ({variety:.2f}/5)                                       │
└─────────────────────────────────────────────────────────────────────┘
"""

# Calculate values
data['Q4_num'] = pd.to_numeric(data['Q4'], errors='coerce')
q11_counts = data['Q11'].value_counts()
local_chain_only = data[data['Q17'].isin(['Local', 'Chain'])]['Q17']

print(summary_stats.format(
    age=data['Q30_num'].mean(),
    orders=data['Q4_num'].median(),
    transport=(data['Q36'] == 'Yes').sum() / len(data) * 100,
    local_pct=(local_chain_only == 'Local').sum() / len(local_chain_only) * 100,
    dominos_pct=(data['Q28'] == "Domino's Pizza").sum() / len(data) * 100,
    mean_visits=data['Q4_num'].mean(),
    med_visits=data['Q4_num'].median(),
    taste=data['Q5_1'].map(importance_map).mean(),
    balance=data['Q5_4'].map(importance_map).mean(),
    crust=data['Q5_3'].map(importance_map).mean(),
    pickup_pct=q11_counts.get('Pick up', 0) / q11_counts.sum() * 100,
    delivery_pct=q11_counts.get('Delivery', 0) / q11_counts.sum() * 100,
    app_pct=q11_counts.get('Third party food delivery app (Uber Eats, Slice, etc.)', 0) / q11_counts.sum() * 100,
    price=data['Q5_7'].map(importance_map).mean(),
    conv=data['Q5_8'].map(importance_map).mean(),
    variety=data['Q8'].map(importance_map).mean()
))

# =============================================================================
# EXPORT SUMMARY TABLE AS CSV
# =============================================================================
summary_table = pd.DataFrame({
    'Metric': [
        'Total Responses', 'Mean Age', 'Median Orders/Month',
        'Prefer Local (%)', 'Prefer Chain (%)', 'Top Restaurant',
        'Prefer Pickup (%)', 'Prefer Delivery (%)',
        'Taste Importance (1-5)', 'Price Importance (1-5)',
        'Convenience Importance (1-5)', 'Variety Importance (1-5)'
    ],
    'Value': [
        161,
        round(data['Q30_num'].mean(), 1),
        int(data['Q4_num'].median()),
        round((local_chain_only == 'Local').sum() / len(local_chain_only) * 100, 1),
        round((local_chain_only == 'Chain').sum() / len(local_chain_only) * 100, 1),
        "Domino's Pizza",
        round(q11_counts.get('Pick up', 0) / q11_counts.sum() * 100, 1),
        round(q11_counts.get('Delivery', 0) / q11_counts.sum() * 100, 1),
        round(data['Q5_1'].map(importance_map).mean(), 2),
        round(data['Q5_7'].map(importance_map).mean(), 2),
        round(data['Q5_8'].map(importance_map).mean(), 2),
        round(data['Q8'].map(importance_map).mean(), 2)
    ]
})
summary_table.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)
print("✓ Generated: summary_statistics.csv")

# =============================================================================
# TOP RESTAURANTS DETAILED TABLE
# =============================================================================
q28_full = data['Q28'].value_counts()
restaurant_table = pd.DataFrame({
    'Rank': range(1, len(q28_full) + 1),
    'Restaurant': q28_full.index,
    'Votes': q28_full.values,
    'Percentage': (q28_full.values / q28_full.sum() * 100).round(1)
})
restaurant_table.to_csv(OUTPUT_DIR / 'restaurant_rankings.csv', index=False)
print("✓ Generated: restaurant_rankings.csv")

print("\n" + "=" * 70)
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print("=" * 70)
