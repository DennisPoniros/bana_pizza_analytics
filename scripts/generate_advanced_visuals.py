"""
BANA255 Pizza Survey - Advanced Seaborn Visualizations
=======================================================
Creates presentation-quality plots with deep insights for stakeholders.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("talk")  # Larger fonts for presentations

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

print("=" * 70)
print("GENERATING ADVANCED VISUALIZATIONS FOR PRESENTATIONS")
print("=" * 70)

# Mappings
importance_map = {
    'Not at all important': 1, 'Slightly important': 2,
    'Moderately important': 3, 'Very important': 4, 'Extremely important': 5
}

loyalty_map = {
    'Not loyal': 1, 'Slightly loyal': 2, 'Moderately loyal': 3,
    'Very loyal': 4, 'Extremely loyal': 5
}

# Convert columns
q5_cols = ['Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5', 'Q5_6', 'Q5_7', 'Q5_8', 'Q5_9']
q5_labels = ['Taste', 'Ingredients', 'Crust', 'Balance', 'Freshness',
             'Appearance', 'Price', 'Convenience', 'Special Features']

for col, label in zip(q5_cols, q5_labels):
    data[f'imp_{label}'] = data[col].map(importance_map)

data['loyalty'] = data['Q29'].map(loyalty_map)
data['orders_month'] = pd.to_numeric(data['Q4'], errors='coerce')
data['expected_price'] = pd.to_numeric(data['Q21_1'], errors='coerce')
data['max_price'] = pd.to_numeric(data['Q21_2'], errors='coerce')
data['age'] = pd.to_numeric(data['Q30'], errors='coerce')

# Create key categorical variables
local_restaurants = ["Joe's Brooklyn Pizza", "Salvatore's Pizza", "Mark's Pizzeria",
                     "Pontillo's Pizza", "Perri's Pizza", "Fire Crust Pizza",
                     "Peels on Wheels", "Brandani's Pizza", "Tony Pepperoni", "Pizza Wizard"]
chain_restaurants = ["Domino's Pizza", "Papa John's", "Little Caesars",
                     "Pizza Hut", "Costco Pizza", "Blaze Pizza"]

data['chose_type'] = data['Q28'].apply(lambda x: 'Local' if x in local_restaurants
                                        else ('Chain' if x in chain_restaurants else 'Other'))
data['stated_pref'] = data['Q17'].apply(lambda x: x if x in ['Local', 'Chain'] else 'Unsure')
data['order_method'] = data['Q11'].apply(lambda x: 'Pickup' if x == 'Pick up'
                                          else ('Delivery' if x == 'Delivery' else 'Third-party'))

# =============================================================================
# FIGURE 18: Correlation Heatmap of Quality Factors
# =============================================================================
print("\nGenerating fig18: Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(12, 10))

# Create correlation matrix for importance factors
imp_cols = [f'imp_{label}' for label in q5_labels]
corr_matrix = data[imp_cols].corr()

# Rename for cleaner display
corr_matrix.index = q5_labels
corr_matrix.columns = q5_labels

# Create heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-0.3, vmax=0.7, ax=ax)

ax.set_title('Quality Factor Correlations\nWhat Priorities Go Together?',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig18_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig18_correlation_heatmap.png")

# =============================================================================
# FIGURE 19: Local vs Chain Choosers - Factor Comparison (Violin Plot)
# =============================================================================
print("Generating fig19: Local vs Chain Factor Comparison...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

key_factors = ['Taste', 'Price', 'Convenience', 'Crust', 'Freshness', 'Balance']
colors = {'Local': '#2ecc71', 'Chain': '#e74c3c'}

for i, factor in enumerate(key_factors):
    plot_data = data[data['chose_type'].isin(['Local', 'Chain'])].copy()

    sns.violinplot(data=plot_data, x='chose_type', y=f'imp_{factor}',
                   hue='chose_type', palette=colors, ax=axes[i], inner='box', legend=False)

    # Add mean markers
    means = plot_data.groupby('chose_type')[f'imp_{factor}'].mean()
    for j, (group, mean) in enumerate(means.items()):
        axes[i].scatter([j], [mean], color='white', s=100, zorder=5, edgecolors='black')
        axes[i].annotate(f'{mean:.2f}', (j, mean), textcoords="offset points",
                        xytext=(15, 0), fontsize=10, fontweight='bold')

    axes[i].set_xlabel('')
    axes[i].set_ylabel('Importance (1-5)')
    axes[i].set_title(f'{factor} Importance', fontweight='bold')
    axes[i].set_ylim(0.5, 5.5)

plt.suptitle('What Local vs Chain Choosers Value\n(White dots = means)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig19_local_chain_violin.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig19_local_chain_violin.png")

# =============================================================================
# FIGURE 20: The Paradox Visualized - Sankey-style Flow
# =============================================================================
print("Generating fig20: Stated vs Actual Behavior Flow...")

fig, ax = plt.subplots(figsize=(14, 8))

# Create flow data
flow_data = data[data['stated_pref'].isin(['Local', 'Chain']) &
                 data['chose_type'].isin(['Local', 'Chain'])].copy()

# Count flows
flows = flow_data.groupby(['stated_pref', 'chose_type']).size().reset_index(name='count')

# Create grouped bar chart showing the paradox
paradox_data = pd.DataFrame({
    'Category': ['Say "Local"\n& Choose Local', 'Say "Local"\n& Choose Chain',
                 'Say "Chain"\n& Choose Local', 'Say "Chain"\n& Choose Chain'],
    'Count': [
        len(flow_data[(flow_data['stated_pref']=='Local') & (flow_data['chose_type']=='Local')]),
        len(flow_data[(flow_data['stated_pref']=='Local') & (flow_data['chose_type']=='Chain')]),
        len(flow_data[(flow_data['stated_pref']=='Chain') & (flow_data['chose_type']=='Local')]),
        len(flow_data[(flow_data['stated_pref']=='Chain') & (flow_data['chose_type']=='Chain')])
    ],
    'Match': ['Match', 'PARADOX', 'Paradox', 'Match']
})

colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
bars = ax.bar(paradox_data['Category'], paradox_data['Count'], color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, count in zip(bars, paradox_data['Count']):
    pct = count / paradox_data['Count'].sum() * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{count}\n({pct:.0f}%)', ha='center', fontsize=12, fontweight='bold')

# Highlight the paradox
ax.annotate('THE PARADOX:\n46 students say "Local"\nbut choose chains!',
            xy=(1, paradox_data['Count'].iloc[1]), xytext=(2.2, paradox_data['Count'].iloc[1] + 15),
            fontsize=12, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

ax.set_ylabel('Number of Students', fontsize=12)
ax.set_title('The Local-Chain Paradox: Words vs Actions\n', fontsize=16, fontweight='bold')
ax.set_ylim(0, max(paradox_data['Count']) * 1.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig20_paradox_flow.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig20_paradox_flow.png")

# =============================================================================
# FIGURE 21: Price Sensitivity vs Taste Importance (Scatter with Regression)
# =============================================================================
print("Generating fig21: Price vs Taste Trade-off...")

fig, ax = plt.subplots(figsize=(12, 8))

plot_data = data[data['chose_type'].isin(['Local', 'Chain'])].copy()

# Create scatter plot with regression lines
sns.scatterplot(data=plot_data, x='imp_Price', y='imp_Taste',
                hue='chose_type', style='chose_type',
                palette={'Local': '#2ecc71', 'Chain': '#e74c3c'},
                s=150, alpha=0.7, ax=ax)

# Add regression lines for each group
for group, color in [('Local', '#2ecc71'), ('Chain', '#e74c3c')]:
    group_data = plot_data[plot_data['chose_type'] == group]
    sns.regplot(data=group_data, x='imp_Price', y='imp_Taste',
                scatter=False, color=color, ax=ax, line_kws={'linewidth': 2})

# Add quadrant lines at means
ax.axhline(y=plot_data['imp_Taste'].mean(), color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=plot_data['imp_Price'].mean(), color='gray', linestyle='--', alpha=0.5)

# Label quadrants
ax.text(1.5, 4.8, 'Low Price Focus\nHigh Taste Focus', fontsize=10, style='italic', alpha=0.7)
ax.text(4.5, 4.8, 'High Price Focus\nHigh Taste Focus', fontsize=10, style='italic', alpha=0.7)
ax.text(1.5, 1.5, 'Low Price Focus\nLow Taste Focus', fontsize=10, style='italic', alpha=0.7)
ax.text(4.5, 1.5, 'High Price Focus\nLow Taste Focus', fontsize=10, style='italic', alpha=0.7)

ax.set_xlabel('Price Importance (1-5)', fontsize=12)
ax.set_ylabel('Taste Importance (1-5)', fontsize=12)
ax.set_title('Price vs Taste: Where Do Customers Fall?\n(with regression lines by choice type)',
             fontsize=14, fontweight='bold')
ax.legend(title='Chose', loc='lower right')
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig21_price_taste_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig21_price_taste_scatter.png")

# =============================================================================
# FIGURE 22: Customer Segments by Order Frequency and Loyalty (Joint Plot)
# =============================================================================
print("Generating fig22: Order Frequency vs Loyalty Joint Plot...")

plot_data = data[['orders_month', 'loyalty', 'chose_type']].dropna()
plot_data = plot_data[plot_data['chose_type'].isin(['Local', 'Chain'])]

g = sns.jointplot(data=plot_data, x='orders_month', y='loyalty',
                  hue='chose_type', kind='scatter',
                  palette={'Local': '#2ecc71', 'Chain': '#e74c3c'},
                  height=10, ratio=4, marginal_kws=dict(fill=True, alpha=0.5))

g.ax_joint.set_xlabel('Orders per Month', fontsize=12)
g.ax_joint.set_ylabel('Loyalty Score (1-5)', fontsize=12)
g.figure.suptitle('Customer Engagement: Frequency vs Loyalty\nby Restaurant Choice Type',
                  fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig22_frequency_loyalty_joint.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig22_frequency_loyalty_joint.png")

# =============================================================================
# FIGURE 23: The Persuadable Segment Deep Dive
# =============================================================================
print("Generating fig23: Persuadable Segment Profile...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Define segments
data['segment'] = 'Other'
data.loc[(data['stated_pref'] == 'Local') & (data['chose_type'] == 'Local'), 'segment'] = 'Loyal Local'
data.loc[(data['stated_pref'] == 'Local') & (data['chose_type'] == 'Chain'), 'segment'] = 'PERSUADABLE'
data.loc[(data['stated_pref'] == 'Chain') & (data['chose_type'] == 'Chain'), 'segment'] = 'Loyal Chain'
data.loc[(data['stated_pref'] == 'Chain') & (data['chose_type'] == 'Local'), 'segment'] = 'Converted'

segment_data = data[data['segment'].isin(['Loyal Local', 'PERSUADABLE', 'Loyal Chain'])]
segment_colors = {'Loyal Local': '#2ecc71', 'PERSUADABLE': '#e74c3c', 'Loyal Chain': '#3498db'}

# Plot 1: Expected Price by Segment
sns.boxplot(data=segment_data, x='segment', y='expected_price',
            hue='segment', palette=segment_colors, ax=axes[0, 0], order=['Loyal Local', 'PERSUADABLE', 'Loyal Chain'], legend=False)
axes[0, 0].set_title('Expected Price by Segment', fontweight='bold')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Expected Price ($)')

# Plot 2: Order Frequency by Segment
sns.boxplot(data=segment_data, x='segment', y='orders_month',
            hue='segment', palette=segment_colors, ax=axes[0, 1], order=['Loyal Local', 'PERSUADABLE', 'Loyal Chain'], legend=False)
axes[0, 1].set_title('Order Frequency by Segment', fontweight='bold')
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Orders per Month')

# Plot 3: Taste Importance by Segment
sns.boxplot(data=segment_data, x='segment', y='imp_Taste',
            hue='segment', palette=segment_colors, ax=axes[1, 0], order=['Loyal Local', 'PERSUADABLE', 'Loyal Chain'], legend=False)
axes[1, 0].set_title('Taste Importance by Segment', fontweight='bold')
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Taste Importance (1-5)')

# Plot 4: Convenience Importance by Segment
sns.boxplot(data=segment_data, x='segment', y='imp_Convenience',
            hue='segment', palette=segment_colors, ax=axes[1, 1], order=['Loyal Local', 'PERSUADABLE', 'Loyal Chain'], legend=False)
axes[1, 1].set_title('Convenience Importance by Segment', fontweight='bold')
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Convenience Importance (1-5)')

plt.suptitle('The PERSUADABLE Segment: Profile Analysis\n(Those who say "Local" but choose chains)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig23_persuadable_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig23_persuadable_profile.png")

# =============================================================================
# FIGURE 24: Time Expectations - Expected vs Willing to Wait
# =============================================================================
print("Generating fig24: Time Expectations Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prepare time data
data['exp_delivery'] = pd.to_numeric(data['Q14_1'], errors='coerce')
data['exp_pickup'] = pd.to_numeric(data['Q14_2'], errors='coerce')
data['willing_delivery'] = pd.to_numeric(data['Q15_1'], errors='coerce')
data['willing_pickup'] = pd.to_numeric(data['Q15_2'], errors='coerce')

# Plot 1: Delivery - Expected vs Willing
delivery_data = data[['exp_delivery', 'willing_delivery']].dropna()
delivery_melted = pd.melt(delivery_data, var_name='Type', value_name='Minutes')
delivery_melted['Type'] = delivery_melted['Type'].map({
    'exp_delivery': 'Expected', 'willing_delivery': 'Willing for Best'
})

sns.violinplot(data=delivery_melted, x='Type', y='Minutes',
               hue='Type', palette=['#3498db', '#2ecc71'], ax=axes[0], inner='box', legend=False)
axes[0].set_title('DELIVERY: Expected vs Willing to Wait', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Minutes')

# Add mean difference annotation
exp_mean = delivery_data['exp_delivery'].mean()
willing_mean = delivery_data['willing_delivery'].mean()
axes[0].annotate(f'+{willing_mean - exp_mean:.0f} min\nfor quality!',
                xy=(1, willing_mean), xytext=(1.3, willing_mean),
                fontsize=11, fontweight='bold', color='#2ecc71',
                arrowprops=dict(arrowstyle='->', color='#2ecc71'))

# Plot 2: Pickup - Expected vs Willing
pickup_data = data[['exp_pickup', 'willing_pickup']].dropna()
pickup_melted = pd.melt(pickup_data, var_name='Type', value_name='Minutes')
pickup_melted['Type'] = pickup_melted['Type'].map({
    'exp_pickup': 'Expected', 'willing_pickup': 'Willing for Best'
})

sns.violinplot(data=pickup_melted, x='Type', y='Minutes',
               hue='Type', palette=['#3498db', '#2ecc71'], ax=axes[1], inner='box', legend=False)
axes[1].set_title('PICKUP: Expected vs Willing to Drive', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Minutes')

# Add mean difference annotation
exp_mean = pickup_data['exp_pickup'].mean()
willing_mean = pickup_data['willing_pickup'].mean()
axes[1].annotate(f'+{willing_mean - exp_mean:.0f} min\nfor quality!',
                xy=(1, willing_mean), xytext=(1.3, willing_mean),
                fontsize=11, fontweight='bold', color='#2ecc71',
                arrowprops=dict(arrowstyle='->', color='#2ecc71'))

plt.suptitle('Quality Premium: How Much Extra Time Will Students Invest?\n',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig24_time_expectations.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig24_time_expectations.png")

# =============================================================================
# FIGURE 25: Domino's Vulnerability Analysis
# =============================================================================
print("Generating fig25: Domino's Vulnerability Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Filter Domino's vs others
dominos = data[data['Q28'] == "Domino's Pizza"].copy()
top_local = data[data['Q28'] == "Joe's Brooklyn Pizza"].copy()
all_others = data[data['Q28'] != "Domino's Pizza"].copy()

# Plot 1: Loyalty Distribution
loyalty_compare = pd.DataFrame({
    'Group': ['Domino\'s'] * len(dominos) + ['Joe\'s Brooklyn'] * len(top_local) + ['All Others'] * len(all_others),
    'Loyalty': list(dominos['loyalty']) + list(top_local['loyalty']) + list(all_others['loyalty'])
})
loyalty_compare = loyalty_compare.dropna()

sns.violinplot(data=loyalty_compare, x='Group', y='Loyalty',
               hue='Group', palette=['#e74c3c', '#2ecc71', '#95a5a6'], ax=axes[0], inner='box', legend=False)
axes[0].set_title('Loyalty Scores\n(Lower = Easier to Poach)', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Loyalty (1-5)')

# Add mean annotations
for i, group in enumerate(['Domino\'s', 'Joe\'s Brooklyn', 'All Others']):
    mean_val = loyalty_compare[loyalty_compare['Group'] == group]['Loyalty'].mean()
    axes[0].annotate(f'μ={mean_val:.2f}', (i, mean_val + 0.3), ha='center', fontweight='bold')

# Plot 2: Stated Preference of Domino's Customers
dom_pref = dominos['stated_pref'].value_counts()
colors = ['#2ecc71' if x == 'Local' else '#e74c3c' if x == 'Chain' else '#95a5a6'
          for x in dom_pref.index]
wedges, texts, autotexts = axes[1].pie(dom_pref.values, labels=dom_pref.index,
                                        autopct='%1.0f%%', colors=colors,
                                        explode=[0.05 if x == 'Local' else 0 for x in dom_pref.index])
axes[1].set_title("Domino's Customers:\nStated Preference", fontweight='bold')

# Highlight the opportunity
axes[1].annotate(f'{dom_pref.get("Local", 0)} customers\nwant LOCAL!',
                xy=(0.5, -0.1), fontsize=11, fontweight='bold', color='#2ecc71',
                ha='center')

# Plot 3: Key Metrics Comparison
metrics = ['imp_Taste', 'imp_Price', 'imp_Convenience', 'loyalty']
metric_labels = ['Taste Imp.', 'Price Imp.', 'Convenience Imp.', 'Loyalty']

dom_means = [dominos[m].mean() for m in metrics]
local_means = [top_local[m].mean() for m in metrics]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[2].bar(x - width/2, dom_means, width, label="Domino's", color='#e74c3c')
bars2 = axes[2].bar(x + width/2, local_means, width, label="Joe's Brooklyn", color='#2ecc71')

axes[2].set_xticks(x)
axes[2].set_xticklabels(metric_labels, rotation=15)
axes[2].set_ylabel('Score')
axes[2].set_title('Key Metrics:\nDomino\'s vs Top Local', fontweight='bold')
axes[2].legend()
axes[2].set_ylim(0, 5.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[2].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.suptitle("Domino's Vulnerability: Why They're Beatable\n", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig25_dominos_vulnerability.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig25_dominos_vulnerability.png")

# =============================================================================
# FIGURE 26: Factor Importance Radar Chart
# =============================================================================
print("Generating fig26: Factor Importance Radar Chart...")

from math import pi

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Prepare data for radar chart
categories = q5_labels
N = len(categories)

# Calculate means for each group
local_data = data[data['chose_type'] == 'Local']
chain_data = data[data['chose_type'] == 'Chain']

local_means = [local_data[f'imp_{cat}'].mean() for cat in categories]
chain_means = [chain_data[f'imp_{cat}'].mean() for cat in categories]

# Normalize to 0-1 scale for radar
local_norm = [(x - 1) / 4 for x in local_means]
chain_norm = [(x - 1) / 4 for x in chain_means]

# Complete the loop
local_norm += local_norm[:1]
chain_norm += chain_norm[:1]

# Calculate angles
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Plot
ax.plot(angles, local_norm, 'o-', linewidth=2, label='Local Choosers', color='#2ecc71')
ax.fill(angles, local_norm, alpha=0.25, color='#2ecc71')
ax.plot(angles, chain_norm, 'o-', linewidth=2, label='Chain Choosers', color='#e74c3c')
ax.fill(angles, chain_norm, alpha=0.25, color='#e74c3c')

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1)

ax.set_title('Quality Factor Priorities:\nLocal vs Chain Choosers', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig26_radar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig26_radar_comparison.png")

# =============================================================================
# FIGURE 27: Price Willingness Distribution by Segment
# =============================================================================
print("Generating fig27: Price Willingness Analysis...")

fig, ax = plt.subplots(figsize=(12, 7))

# Create price data with segments
price_data = data[['expected_price', 'max_price', 'chose_type']].dropna()
price_data = price_data[price_data['chose_type'].isin(['Local', 'Chain'])]

# KDE plot for expected vs max price
for group, color in [('Local', '#2ecc71'), ('Chain', '#e74c3c')]:
    group_data = price_data[price_data['chose_type'] == group]

    # Expected price
    sns.kdeplot(data=group_data, x='expected_price', ax=ax,
                color=color, linestyle='-', linewidth=2, label=f'{group} - Expected')
    # Max price
    sns.kdeplot(data=group_data, x='max_price', ax=ax,
                color=color, linestyle='--', linewidth=2, label=f'{group} - Max')

# Add vertical lines for means
local_exp = price_data[price_data['chose_type']=='Local']['expected_price'].mean()
local_max = price_data[price_data['chose_type']=='Local']['max_price'].mean()
chain_exp = price_data[price_data['chose_type']=='Chain']['expected_price'].mean()
chain_max = price_data[price_data['chose_type']=='Chain']['max_price'].mean()

ax.axvline(local_exp, color='#2ecc71', linestyle=':', alpha=0.7)
ax.axvline(chain_exp, color='#e74c3c', linestyle=':', alpha=0.7)

# Highlight the "sweet spot"
ax.axvspan(17, 21, alpha=0.2, color='gold', label='Sweet Spot ($17-21)')

ax.set_xlabel('Price ($)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Price Expectations: Expected vs Maximum Willingness to Pay\n(Sweet spot highlighted)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.set_xlim(5, 40)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig27_price_willingness.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig27_price_willingness.png")

# =============================================================================
# FIGURE 28: Demographics Breakdown by Choice Type
# =============================================================================
print("Generating fig28: Demographics by Choice Type...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

segment_data = data[data['chose_type'].isin(['Local', 'Chain'])]

# Plot 1: Year in School
year_order = ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Super Senior', 'Graduate student']
year_data = segment_data.groupby(['Q31', 'chose_type']).size().unstack(fill_value=0)
year_data = year_data.reindex(year_order).dropna()

year_data.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71'], width=0.8)
axes[0, 0].set_title('Year in School', fontweight='bold')
axes[0, 0].set_xlabel('')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
axes[0, 0].legend(title='Chose')

# Plot 2: Residence
residence_data = segment_data.groupby(['Q33', 'chose_type']).size().unstack(fill_value=0)
residence_data.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#2ecc71'], width=0.8)
axes[0, 1].set_title('Residence', fontweight='bold')
axes[0, 1].set_xlabel('')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
axes[0, 1].legend(title='Chose')

# Plot 3: Has Transportation
transport_data = segment_data.groupby(['Q36', 'chose_type']).size().unstack(fill_value=0)
transport_pct = transport_data.div(transport_data.sum(axis=1), axis=0) * 100
transport_pct.plot(kind='bar', stacked=True, ax=axes[1, 0],
                   color=['#e74c3c', '#2ecc71'], width=0.6)
axes[1, 0].set_title('Has Transportation', fontweight='bold')
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].set_xticklabels(['No', 'Yes'], rotation=0)
axes[1, 0].legend(title='Chose', loc='center right')

# Plot 4: Order Method
method_data = segment_data.groupby(['order_method', 'chose_type']).size().unstack(fill_value=0)
method_data.plot(kind='bar', ax=axes[1, 1], color=['#e74c3c', '#2ecc71'], width=0.8)
axes[1, 1].set_title('Order Method Preference', fontweight='bold')
axes[1, 1].set_xlabel('')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
axes[1, 1].legend(title='Chose')

plt.suptitle('Who Chooses Local vs Chain?\nDemographic Breakdown', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig28_demographics_choice.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig28_demographics_choice.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("ADVANCED VISUALIZATIONS COMPLETE")
print("=" * 70)
print(f"""
Generated 11 new presentation-quality figures:

  fig18_correlation_heatmap.png     - Quality factor correlations
  fig19_local_chain_violin.png      - Factor comparison by choice type
  fig20_paradox_flow.png            - Stated vs actual behavior flow
  fig21_price_taste_scatter.png     - Price vs taste trade-off
  fig22_frequency_loyalty_joint.png - Order frequency vs loyalty
  fig23_persuadable_profile.png     - Persuadable segment deep dive
  fig24_time_expectations.png       - Expected vs willing time investment
  fig25_dominos_vulnerability.png   - Competitor vulnerability analysis
  fig26_radar_comparison.png        - Factor priorities radar chart
  fig27_price_willingness.png       - Price distribution analysis
  fig28_demographics_choice.png     - Demographics by choice type

All figures saved to: {OUTPUT_DIR.absolute()}
""")
