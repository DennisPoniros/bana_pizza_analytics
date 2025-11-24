"""
BANA255 Pizza Survey - Competitive Model Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

# Mappings
importance_map = {
    'Not at all important': 1, 'Slightly important': 2,
    'Moderately important': 3, 'Very important': 4, 'Extremely important': 5
}

q5_cols = ['Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5', 'Q5_6', 'Q5_7', 'Q5_8', 'Q5_9']
labels = ['Taste', 'Ingredients', 'Crust', 'Balance', 'Freshness',
          'Appearance', 'Price', 'Convenience', 'Special Features']

for col in q5_cols:
    data[f'{col}_score'] = data[col].map(importance_map)

data['loyalty_score'] = data['Q29'].map({
    'Not loyal': 1, 'Slightly loyal': 2, 'Moderately loyal': 3,
    'Very loyal': 4, 'Extremely loyal': 5
})

# =============================================================================
# FIGURE 7: Weighted Importance Model - "Best Pizza" Definition
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

weights = []
for col, label in zip(q5_cols, labels):
    weights.append({
        'Factor': label,
        'Mean': data[f'{col}_score'].mean(),
        'Pct_High': ((data[f'{col}_score'] >= 4).sum() / len(data)) * 100
    })

weights_df = pd.DataFrame(weights).sort_values('Mean', ascending=True)
total = weights_df['Mean'].sum()
weights_df['Normalized'] = weights_df['Mean'] / total * 100

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(weights_df)))

bars = ax.barh(weights_df['Factor'], weights_df['Normalized'], color=colors)
ax.set_xlabel('Weight in "Best Pizza" Definition (%)')
ax.set_title('Weighted Importance Model: What Defines "The Best" Pizza?\n(Normalized weights sum to 100%)',
             fontsize=12, fontweight='bold')

for bar, (_, row) in zip(bars, weights_df.iterrows()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{row["Normalized"]:.1f}%', va='center', fontsize=9)

ax.set_xlim(0, 18)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_importance_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig7_importance_weights.png")

# =============================================================================
# FIGURE 8: Competitive Threat Ranking
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate threat scores (simplified from competitive_model.py)
restaurants = data['Q28'].value_counts().head(10).index.tolist()
threat_scores = []

max_share = data['Q28'].value_counts().max() / len(data) * 100

for rest in restaurants:
    rest_data = data[data['Q28'] == rest]
    n = len(rest_data)
    share = n / len(data) * 100

    # Loyalty
    loyalty = rest_data['loyalty_score'].mean()
    loyalty_score = ((loyalty - 1) / 4) * 100 if not np.isnan(loyalty) else 50

    # Share score
    share_score = (share / max_share) * 100

    # Local capture
    lp = rest_data[rest_data['Q17'].isin(['Local', 'Chain'])]['Q17']
    local_capture = (lp == 'Local').sum() / len(lp) * 100 if len(lp) > 0 else 50

    # Type
    is_chain = rest in ["Domino's Pizza", "Papa John's", "Little Caesars",
                        "Pizza Hut", "Costco Pizza", "Blaze Pizza"]

    composite = share_score * 0.35 + loyalty_score * 0.25 + local_capture * 0.25 + 15  # base

    threat_scores.append({
        'Restaurant': rest,
        'Score': composite,
        'Type': 'Chain' if is_chain else 'Local',
        'Share': share
    })

threat_df = pd.DataFrame(threat_scores).sort_values('Score', ascending=True)

colors = ['#e74c3c' if t == 'Chain' else '#2ecc71' for t in threat_df['Type']]
bars = ax.barh(threat_df['Restaurant'], threat_df['Score'], color=colors)

ax.set_xlabel('Competitive Threat Score')
ax.set_title('Competitive Threat Ranking for New Local Entrant\n(Higher = More Threatening)',
             fontsize=12, fontweight='bold')

for bar, score in zip(bars, threat_df['Score']):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{score:.0f}', va='center', fontsize=10)

# Legend
chain_patch = mpatches.Patch(color='#e74c3c', label='Chain')
local_patch = mpatches.Patch(color='#2ecc71', label='Local')
ax.legend(handles=[chain_patch, local_patch], loc='lower right')

ax.set_xlim(0, 90)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig8_competitive_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig8_competitive_ranking.png")

# =============================================================================
# FIGURE 9: The Local-Chain Paradox
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Stated preference
q17_counts = data['Q17'].value_counts()
axes[0].pie(q17_counts.values, labels=q17_counts.index, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c', '#95a5a6'], explode=(0.05, 0, 0))
axes[0].set_title('Stated Preference:\n"Do you prefer local or chain?"', fontsize=11, fontweight='bold')

# Right: Actual behavior
actual = data['Q28'].value_counts().head(8)
is_chain = [r in ["Domino's Pizza", "Papa John's", "Little Caesars",
                   "Pizza Hut", "Costco Pizza", "Blaze Pizza"] for r in actual.index]
colors = ['#e74c3c' if c else '#2ecc71' for c in is_chain]

bars = axes[1].barh(actual.index[::-1], actual.values[::-1], color=colors[::-1])
axes[1].set_xlabel('Number of Votes')
axes[1].set_title('Actual Behavior:\nFavorite Pizza Place', fontsize=11, fontweight='bold')

# Add legend
chain_patch = mpatches.Patch(color='#e74c3c', label='Chain')
local_patch = mpatches.Patch(color='#2ecc71', label='Local')
axes[1].legend(handles=[chain_patch, local_patch], loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig9_local_chain_paradox.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig9_local_chain_paradox.png")

# =============================================================================
# FIGURE 10: Domino's Customer Profile (The Opportunity)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

dominos = data[data['Q28'] == "Domino's Pizza"]
others = data[data['Q28'] != "Domino's Pizza"]

# Left: Domino's customers' stated preference
dom_pref = dominos['Q17'].value_counts()
axes[0].pie(dom_pref.values, labels=dom_pref.index, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c', '#95a5a6'])
axes[0].set_title("Domino's Customers:\nStated Local vs Chain Preference", fontsize=11, fontweight='bold')

# Right: Key metrics comparison
metrics = ['Loyalty\n(1-5)', 'Expected\nPrice ($)', 'Orders/\nMonth']
dom_vals = [dominos['loyalty_score'].mean(),
            pd.to_numeric(dominos['Q21_1'], errors='coerce').mean(),
            pd.to_numeric(dominos['Q4'], errors='coerce').mean()]
other_vals = [others['loyalty_score'].mean(),
              pd.to_numeric(others['Q21_1'], errors='coerce').mean(),
              pd.to_numeric(others['Q4'], errors='coerce').mean()]

# Normalize for comparison (different scales)
dom_norm = [dom_vals[0]/5*100, dom_vals[1]/25*100, dom_vals[2]/5*100]
other_norm = [other_vals[0]/5*100, other_vals[1]/25*100, other_vals[2]/5*100]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[1].bar(x - width/2, dom_norm, width, label="Domino's", color='#3498db')
bars2 = axes[1].bar(x + width/2, other_norm, width, label='Others', color='#95a5a6')

axes[1].set_ylabel('Normalized Score')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics)
axes[1].legend()
axes[1].set_title("Domino's vs Others: Key Metrics\n(Lower loyalty = opportunity!)", fontsize=11, fontweight='bold')

# Add actual values as text
for bar, val in zip(bars1, dom_vals):
    if val > 10:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'${val:.0f}', ha='center', fontsize=9)
    else:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig10_dominos_opportunity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig10_dominos_opportunity.png")

# =============================================================================
# FIGURE 11: Strategic Positioning Map
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate price expectation and taste importance for each restaurant
restaurants = data['Q28'].value_counts().head(10).index.tolist()
plot_data = []

for rest in restaurants:
    rest_data = data[data['Q28'] == rest]
    n = len(rest_data)

    price_exp = pd.to_numeric(rest_data['Q21_1'], errors='coerce').mean()
    taste_imp = rest_data['Q5_1_score'].mean()
    is_chain = rest in ["Domino's Pizza", "Papa John's", "Little Caesars",
                        "Pizza Hut", "Costco Pizza", "Blaze Pizza"]

    plot_data.append({
        'Restaurant': rest,
        'Price': price_exp,
        'Taste': taste_imp,
        'Size': n,
        'Type': 'Chain' if is_chain else 'Local'
    })

plot_df = pd.DataFrame(plot_data)

# Create scatter plot
for _, row in plot_df.iterrows():
    color = '#e74c3c' if row['Type'] == 'Chain' else '#2ecc71'
    ax.scatter(row['Price'], row['Taste'], s=row['Size']*15, c=color, alpha=0.6, edgecolors='black')
    ax.annotate(row['Restaurant'].replace(' Pizza', '').replace("'s", ''),
                (row['Price'], row['Taste']), fontsize=8,
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Customer Expected Price ($)')
ax.set_ylabel('Customer Taste Importance (1-5)')
ax.set_title('Strategic Positioning Map\n(Bubble size = market share)',
             fontsize=12, fontweight='bold')

# Add quadrant labels
ax.axhline(y=4.4, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=17, color='gray', linestyle='--', alpha=0.5)
ax.text(14.5, 4.65, 'Value\nSeekers', fontsize=9, ha='center', style='italic', alpha=0.7)
ax.text(20, 4.65, 'Premium\nQuality', fontsize=9, ha='center', style='italic', alpha=0.7)
ax.text(14.5, 4.15, 'Budget\nBuyers', fontsize=9, ha='center', style='italic', alpha=0.7)
ax.text(20, 4.15, 'Indifferent', fontsize=9, ha='center', style='italic', alpha=0.7)

# Legend
chain_patch = mpatches.Patch(color='#e74c3c', label='Chain', alpha=0.6)
local_patch = mpatches.Patch(color='#2ecc71', label='Local', alpha=0.6)
ax.legend(handles=[chain_patch, local_patch], loc='upper left')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig11_positioning_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig11_positioning_map.png")

# =============================================================================
# FIGURE 12: Regression Insights - What Predicts Choosing Local
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Data from regression analysis
predictors = ['Crust Importance', 'Orders/Month', 'Expected Price',
              'Prefers Local (stated)', 'Has Transportation']
diff_values = [0.46, 0.95, 2.0, 37.5, 27.9]  # Differences from regression output
p_values = [0.0079, 0.0158, 0.0260, 0.0000, 0.0004]

colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in p_values]

bars = ax.barh(predictors, diff_values, color=colors)
ax.set_xlabel('Difference (Local Choosers - Chain Choosers)')
ax.set_title('What Predicts Choosing a LOCAL Pizza Place?\n(Green = Statistically Significant, p < 0.05)',
             fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)

for bar, p in zip(bars, p_values):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'p={p:.4f}{sig}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig12_local_predictors.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig12_local_predictors.png")

print("\n✓ All competitive model visualizations generated!")
print(f"  Output directory: {OUTPUT_DIR.absolute()}")
