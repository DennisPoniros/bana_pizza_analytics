#!/usr/bin/env python3
"""
Side Order Visualization Generator
===================================
Generates publication-quality figures for side order analysis.

Figures:
- fig47: Side order popularity rankings
- fig48: Side order likelihood distribution
- fig49: Side spending distribution
- fig50: Side preferences by restaurant type
- fig51: Side order correlation heatmap
- fig52: Revenue opportunity matrix

Version: 8.0
Date: 2025-12-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading data...")
df = pd.read_excel('BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx')
df = df.iloc[1:].reset_index(drop=True)
df = df[df['Q2'] == 'Yes'].reset_index(drop=True)
print(f"Analyzing {len(df)} consented respondents")

# Side items mapping
side_items = {
    'Q25_1': 'Fries',
    'Q25_2': 'Salad',
    'Q25_3': 'Garlic Knots',
    'Q25_4': 'Mozz Sticks',
    'Q25_5': 'Wings',
    'Q25_6': 'Calamari',
    'Q25_7': 'Bruschetta',
    'Q25_8': 'Stuffed Mushrooms',
    'Q25_9': 'Onion Rings',
    'Q25_10': 'Cheesecake'
}

# Likert scale mapping
likelihood_map = {
    'Extremely unlikely': 1,
    'Somewhat unlikely': 2,
    'Neither likely nor unlikely': 3,
    'Somewhat likely': 4,
    'Extremely likely': 5
}

# Convert to numeric
for col, name in side_items.items():
    df[col + '_numeric'] = df[col].map(likelihood_map)

df['side_spend'] = pd.to_numeric(df['Q26'], errors='coerce')

# Calculate stats
side_stats = []
for col, name in side_items.items():
    data = df[col + '_numeric'].dropna()
    side_stats.append({
        'Side Item': name,
        'Mean': data.mean(),
        'Median': data.median(),
        '% Likely': (data >= 4).sum() / len(data) * 100,
        '% Definitely': (data == 5).sum() / len(data) * 100,
        'N': len(data)
    })

side_df = pd.DataFrame(side_stats).sort_values('Mean', ascending=False).reset_index(drop=True)

# =============================================================================
# FIGURE 47: Side Order Popularity Rankings
# =============================================================================
print("Generating fig47: Side Order Popularity Rankings...")

fig, ax = plt.subplots(figsize=(12, 7))

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(side_df)))[::-1]
bars = ax.barh(range(len(side_df)), side_df['Mean'], color=colors, edgecolor='black', linewidth=0.5)

ax.set_yticks(range(len(side_df)))
ax.set_yticklabels(side_df['Side Item'], fontsize=11)
ax.set_xlabel('Mean Likelihood Score (1-5 scale)', fontsize=12, fontweight='bold')
ax.set_title('Side Order Popularity Rankings\n"How likely are you to order these sides with pizza?"',
             fontsize=14, fontweight='bold')

# Add value labels
for i, (bar, val, pct) in enumerate(zip(bars, side_df['Mean'], side_df['% Likely'])):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f} ({pct:.0f}% likely)', va='center', fontsize=10)

ax.set_xlim(0, 5.5)
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='Neutral (3.0)')
ax.legend(loc='lower right')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/fig47_side_order_rankings.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: outputs/fig47_side_order_rankings.png")

# =============================================================================
# FIGURE 48: Side Order Likelihood Distribution
# =============================================================================
print("Generating fig48: Side Order Likelihood Distribution...")

fig, axes = plt.subplots(2, 5, figsize=(16, 8))
axes = axes.flatten()

for idx, (col, name) in enumerate(side_items.items()):
    ax = axes[idx]
    data = df[col + '_numeric'].dropna()

    # Count by category
    counts = data.value_counts().sort_index()
    all_cats = range(1, 6)
    counts = counts.reindex(all_cats, fill_value=0)

    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Count' if idx % 5 == 0 else '')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1\nVery\nUnlikely', '2', '3\nNeutral', '4', '5\nVery\nLikely'], fontsize=8)

    # Add mean line
    mean_val = data.mean()
    ax.axvline(x=mean_val, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(mean_val + 0.1, ax.get_ylim()[1] * 0.9, f'μ={mean_val:.2f}', fontsize=9, color='blue')

fig.suptitle('Side Order Likelihood Distributions by Item', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/fig48_side_order_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: outputs/fig48_side_order_distributions.png")

# =============================================================================
# FIGURE 49: Side Spending Distribution
# =============================================================================
print("Generating fig49: Side Spending Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
spend_data = df['side_spend'].dropna()
ax1.hist(spend_data, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(spend_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${spend_data.mean():.2f}')
ax1.axvline(spend_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${spend_data.median():.2f}')
ax1.set_xlabel('Dollars Spent on Sides', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Students', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Side Order Spending', fontsize=13, fontweight='bold')
ax1.legend()

# Box plot with spending segments
ax2 = axes[1]
bins = [0, 5, 10, 15, 20, 100]
labels = ['$0-5\nLow', '$6-10\nMedium', '$11-15\nHigh', '$16-20\nPremium', '$20+\nVery High']
df['spend_segment'] = pd.cut(df['side_spend'], bins=bins, labels=labels, include_lowest=True)
segment_counts = df['spend_segment'].value_counts().sort_index()

colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(segment_counts)))
ax2.bar(range(len(segment_counts)), segment_counts.values, color=colors, edgecolor='black')
ax2.set_xticks(range(len(segment_counts)))
ax2.set_xticklabels(segment_counts.index, fontsize=10)
ax2.set_xlabel('Spending Segment', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Students', fontsize=12, fontweight='bold')
ax2.set_title('Side Spending Segments', fontsize=13, fontweight='bold')

# Add percentages
for i, (count, label) in enumerate(zip(segment_counts.values, segment_counts.index)):
    pct = count / segment_counts.sum() * 100
    ax2.text(i, count + 1, f'{pct:.0f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig49_side_spending.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: outputs/fig49_side_spending.png")

# =============================================================================
# FIGURE 50: Side Preferences by Restaurant Type
# =============================================================================
print("Generating fig50: Side Preferences by Restaurant Type...")

# Classify restaurants
local_keywords = ['Joe\'s Brooklyn', 'Salvatore', 'Mark\'s', 'Pontillo', 'Fiamma']
chain_keywords = ['Domino', 'Papa John', 'Little Caesar', 'Pizza Hut', 'Costco', 'Blaze']

def classify_restaurant(name):
    if pd.isna(name):
        return 'Unknown'
    name_lower = str(name).lower()
    for kw in local_keywords:
        if kw.lower() in name_lower:
            return 'Local'
    for kw in chain_keywords:
        if kw.lower() in name_lower:
            return 'Chain'
    return 'Other'

df['restaurant_type'] = df['Q28'].apply(classify_restaurant)

fig, ax = plt.subplots(figsize=(12, 7))

# Calculate means by restaurant type
local_means = []
chain_means = []
for col, name in side_items.items():
    local_data = df[df['restaurant_type'] == 'Local'][col + '_numeric'].dropna()
    chain_data = df[df['restaurant_type'] == 'Chain'][col + '_numeric'].dropna()
    local_means.append(local_data.mean() if len(local_data) > 0 else np.nan)
    chain_means.append(chain_data.mean() if len(chain_data) > 0 else np.nan)

x = np.arange(len(side_items))
width = 0.35

bars1 = ax.bar(x - width/2, local_means, width, label='Local Choosers', color='forestgreen', edgecolor='black')
bars2 = ax.bar(x + width/2, chain_means, width, label='Chain Choosers', color='royalblue', edgecolor='black')

ax.set_ylabel('Mean Likelihood (1-5)', fontsize=12, fontweight='bold')
ax.set_xlabel('Side Item', fontsize=12, fontweight='bold')
ax.set_title('Side Order Preferences by Restaurant Type', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(list(side_items.values()), rotation=45, ha='right', fontsize=10)
ax.legend()
ax.set_ylim(0, 5)
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/fig50_sides_by_restaurant_type.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: outputs/fig50_sides_by_restaurant_type.png")

# =============================================================================
# FIGURE 51: Side Order Correlation Heatmap
# =============================================================================
print("Generating fig51: Side Order Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

# Correlation matrix
corr_cols = [col + '_numeric' for col in side_items.keys()]
corr_data = df[corr_cols].dropna()
corr_matrix = corr_data.corr(method='spearman')

# Rename for display
corr_matrix.index = list(side_items.values())
corr_matrix.columns = list(side_items.values())

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-0.5, vmax=0.7, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Spearman Correlation'})

ax.set_title('Side Order Preference Correlations\n(Spearman)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig51_side_correlations.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: outputs/fig51_side_correlations.png")

# =============================================================================
# FIGURE 52: Revenue Opportunity Matrix
# =============================================================================
print("Generating fig52: Revenue Opportunity Matrix...")

fig, ax = plt.subplots(figsize=(11, 8))

# Calculate metrics for each side
for col, name in side_items.items():
    data = df[col + '_numeric'].dropna()
    mean_score = data.mean()
    pct_likely = (data >= 4).sum() / len(data) * 100

    # Find index in side_df
    row = side_df[side_df['Side Item'] == name].iloc[0]

    # Plot
    ax.scatter(mean_score, pct_likely, s=200, alpha=0.7)
    ax.annotate(name, (mean_score, pct_likely), textcoords="offset points",
                xytext=(5, 5), fontsize=10, fontweight='bold')

# Add quadrant lines
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5)

# Quadrant labels
ax.text(4.2, 70, 'HIGH PRIORITY\n(High demand, many likely)', fontsize=10,
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.text(2.2, 70, 'NICHE\n(Few like it, but intensely)', fontsize=10,
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax.text(4.2, 15, 'MODERATE\n(Popular but tepid)', fontsize=10,
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(2.2, 15, 'LOW PRIORITY\n(Low demand overall)', fontsize=10,
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

ax.set_xlabel('Mean Likelihood Score (1-5)', fontsize=12, fontweight='bold')
ax.set_ylabel('% Who Would Likely Order (4 or 5)', fontsize=12, fontweight='bold')
ax.set_title('Side Order Revenue Opportunity Matrix', fontsize=14, fontweight='bold')
ax.set_xlim(1.5, 4.8)
ax.set_ylim(0, 85)

plt.tight_layout()
plt.savefig('outputs/fig52_revenue_opportunity.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: outputs/fig52_revenue_opportunity.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 60)
print("""
Generated figures:
  - fig47_side_order_rankings.png     : Horizontal bar chart of popularity
  - fig48_side_order_distributions.png: Distribution histograms by item
  - fig49_side_spending.png           : Spending histogram and segments
  - fig50_sides_by_restaurant_type.png: Local vs Chain comparison
  - fig51_side_correlations.png       : Correlation heatmap
  - fig52_revenue_opportunity.png     : Strategic opportunity matrix
""")
