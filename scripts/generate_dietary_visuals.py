"""
BANA255 Pizza Survey - Dietary Accommodation Visualizations
Generates figures 41-46 for dietary analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

print(f"Loaded {len(data)} valid responses")

# Mapping
importance_map = {
    'Not at all important': 1,
    'Slightly important': 2,
    'Moderately important': 3,
    'Very important': 4,
    'Extremely important': 5
}

q9_cols = ['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5', 'Q9_6', 'Q9_7', 'Q9_8', 'Q9_9', 'Q9_10', 'Q9_11']
q9_short_labels = {
    'Q9_1': 'Gluten-Free',
    'Q9_2': 'Vegan/Dairy-Free',
    'Q9_3': 'Cross-Contam.\nPrevention',
    'Q9_4': 'Low-Carb/Keto',
    'Q9_5': 'Plant-Based\nProtein',
    'Q9_6': 'Allergen\nTransparency',
    'Q9_7': 'Organic/\nNon-GMO',
    'Q9_8': 'Reduced\nSodium',
    'Q9_9': 'Halal/Kosher',
    'Q9_10': 'Half-and-Half\nOption',
    'Q9_11': 'Allergen\nLabeling'
}

# Convert to numeric
for col in q9_cols:
    data[f'{col}_score'] = data[col].map(importance_map)

# =============================================================================
# FIGURE 41: Dietary Importance Rankings (Horizontal Bar Chart)
# =============================================================================
print("Generating fig41_dietary_importance.png...")

# Calculate stats
dietary_stats = []
for col in q9_cols:
    scores = data[f'{col}_score'].dropna()
    dietary_stats.append({
        'factor': q9_short_labels[col],
        'mean': scores.mean(),
        'std': scores.std(),
        'pct_high': ((scores >= 4).sum() / len(scores)) * 100
    })

dietary_df = pd.DataFrame(dietary_stats).sort_values('mean', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(dietary_df)))[::-1]

bars = ax.barh(dietary_df['factor'], dietary_df['mean'], color=colors, edgecolor='white', linewidth=0.5)

# Add value labels
for bar, (_, row) in zip(bars, dietary_df.iterrows()):
    width = bar.get_width()
    ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
            f'{row["mean"]:.2f}', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Mean Importance Score (1-5)', fontsize=12)
ax.set_title('Dietary Accommodation Importance Rankings\n(RIT Pizza Survey, n=161)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 3.5)
ax.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Moderate threshold')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('outputs/fig41_dietary_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/fig41_dietary_importance.png")

# =============================================================================
# FIGURE 42: Dietary Importance Distribution (Stacked Bar)
# =============================================================================
print("Generating fig42_dietary_distribution.png...")

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate distribution for each factor
importance_levels = ['Not at all important', 'Slightly important', 'Moderately important',
                     'Very important', 'Extremely important']
level_colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']

# Sort by mean importance
factor_order = dietary_df['factor'].tolist()[::-1]  # Reverse for top-to-bottom

distributions = []
for col in q9_cols:
    factor_name = q9_short_labels[col]
    dist = {}
    total = len(data[col].dropna())
    for level in importance_levels:
        count = (data[col] == level).sum()
        dist[level] = (count / total) * 100
    dist['factor'] = factor_name.replace('\n', ' ')
    distributions.append(dist)

dist_df = pd.DataFrame(distributions)
dist_df = dist_df.set_index('factor')

# Reorder based on mean importance
mean_order = dietary_df['factor'].str.replace('\n', ' ').tolist()[::-1]
dist_df = dist_df.reindex(mean_order)

# Create stacked horizontal bar chart
left = np.zeros(len(dist_df))
for i, level in enumerate(importance_levels):
    values = dist_df[level].values
    ax.barh(dist_df.index, values, left=left, color=level_colors[i], label=level, edgecolor='white', linewidth=0.3)
    left += values

ax.set_xlabel('Percentage of Respondents (%)', fontsize=12)
ax.set_title('Distribution of Dietary Accommodation Importance Ratings\n(RIT Pizza Survey, n=161)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('outputs/fig42_dietary_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/fig42_dietary_distribution.png")

# =============================================================================
# FIGURE 43: Dietary Consciousness Segments (Pie Chart)
# =============================================================================
print("Generating fig43_dietary_segments.png...")

# Calculate dietary consciousness
score_cols = [f'{col}_score' for col in q9_cols]
data['dietary_consciousness'] = data[score_cols].mean(axis=1)

# Segment
data['dietary_segment'] = pd.cut(
    data['dietary_consciousness'],
    bins=[0, 2.0, 3.0, 5],
    labels=['Dietary Indifferent\n(Score < 2)', 'Moderately Concerned\n(Score 2-3)', 'Dietary Conscious\n(Score 3+)']
)

segment_counts = data['dietary_segment'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
colors_pie = ['#ef8a62', '#f7f7f7', '#67a9cf']
wedges, texts, autotexts = ax1.pie(
    segment_counts.values,
    labels=segment_counts.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    explode=[0.02, 0.02, 0.05],
    startangle=90
)
autotexts[0].set_fontsize(12)
autotexts[0].set_fontweight('bold')
ax1.set_title('Dietary Consciousness Segments', fontsize=14, fontweight='bold')

# Bar chart with segment characteristics
segments = ['Dietary Indifferent\n(Score < 2)', 'Moderately Concerned\n(Score 2-3)', 'Dietary Conscious\n(Score 3+)']
segment_labels = ['Indifferent', 'Moderate', 'Conscious']
metrics = ['Mean Dietary Score', 'Mean Taste Imp.', '% Prefer Local']

seg_data = []
for seg, label in zip(segments, segment_labels):
    seg_subset = data[data['dietary_segment'] == seg]
    seg_data.append({
        'Segment': label,
        'Mean Dietary Score': seg_subset['dietary_consciousness'].mean(),
        'Mean Taste Imp.': seg_subset['Q5_1'].map(importance_map).mean(),
        '% Prefer Local': (seg_subset['Q17'] == 'Local').sum() / len(seg_subset) * 100 if len(seg_subset) > 0 else 0
    })

seg_df = pd.DataFrame(seg_data)

x = np.arange(len(segment_labels))
width = 0.25

bars1 = ax2.bar(x - width, seg_df['Mean Dietary Score'], width, label='Dietary Score', color='#67a9cf')
bars2 = ax2.bar(x, seg_df['Mean Taste Imp.'], width, label='Taste Importance', color='#ef8a62')
bars3 = ax2.bar(x + width, seg_df['% Prefer Local']/20, width, label='% Local Pref (scaled)', color='#91cf60')

ax2.set_ylabel('Score / Scaled Percentage')
ax2.set_title('Segment Characteristics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(segment_labels)
ax2.legend()
ax2.set_ylim(0, 5)

# Add value labels
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}',
             ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/fig43_dietary_segments.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/fig43_dietary_segments.png")

# =============================================================================
# FIGURE 44: Correlation Heatmap - Dietary Factors
# =============================================================================
print("Generating fig44_dietary_correlations.png...")

# Create correlation matrix
corr_data = data[score_cols].dropna()
corr_data.columns = [q9_short_labels[col].replace('\n', ' ') for col in q9_cols]
corr_matrix = corr_data.corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'label': 'Spearman Correlation'})

ax.set_title('Correlation Between Dietary Accommodation Factors\n(Spearman Rank Correlation)',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig44_dietary_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/fig44_dietary_correlations.png")

# =============================================================================
# FIGURE 45: Dietary Importance vs Overall Pizza Quality Importance
# =============================================================================
print("Generating fig45_dietary_vs_quality.png...")

# Map Q5 columns
q5_labels = {
    'Q5_1': 'Taste',
    'Q5_2': 'Ingredients',
    'Q5_3': 'Crust',
    'Q5_4': 'Balance',
    'Q5_5': 'Freshness',
    'Q5_6': 'Appearance',
    'Q5_7': 'Price',
    'Q5_8': 'Convenience',
    'Q5_9': 'Special Features'
}

for col in q5_labels.keys():
    data[f'{col}_score'] = data[col].map(importance_map)

fig, ax = plt.subplots(figsize=(12, 6))

# Calculate means for both categories
dietary_means = [data[f'{col}_score'].mean() for col in q9_cols]
quality_means = [data[f'{col}_score'].mean() for col in q5_labels.keys()]

dietary_avg = np.mean(dietary_means)
quality_avg = np.mean(quality_means)

x = np.arange(2)
bars = ax.bar(x, [quality_avg, dietary_avg], color=['#2166ac', '#b2182b'], width=0.6, edgecolor='white', linewidth=2)

# Add value labels
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['Pizza Quality Factors\n(Taste, Price, etc.)', 'Dietary Accommodation\nFactors'], fontsize=12)
ax.set_ylabel('Mean Importance Score (1-5)', fontsize=12)
ax.set_title('Comparison: Pizza Quality vs Dietary Accommodation Importance\n(RIT Pizza Survey, n=161)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 5)
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Moderate threshold')

# Add annotation
ax.annotate(f'Quality factors rated\n{(quality_avg/dietary_avg - 1)*100:.0f}% higher',
            xy=(0.5, max(quality_avg, dietary_avg) + 0.3), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('outputs/fig45_dietary_vs_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/fig45_dietary_vs_quality.png")

# =============================================================================
# FIGURE 46: Dietary Options Actively Sought (Q10)
# =============================================================================
print("Generating fig46_dietary_options_sought.png...")

# Parse Q10 responses
from collections import Counter

q10_data = data['Q10'].dropna()
all_options = []
for response in q10_data:
    if pd.notna(response) and str(response).strip():
        options = str(response).split(',')
        all_options.extend([opt.strip() for opt in options if opt.strip() and opt.strip() != 'nan'])

option_counts = Counter(all_options)
total_seekers = len(q10_data)

# Filter and sort
filtered_options = {k: v for k, v in option_counts.items() if k not in ['nan', '']}
sorted_options = dict(sorted(filtered_options.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots(figsize=(10, 6))

if sorted_options:
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(sorted_options)))
    bars = ax.barh(list(sorted_options.keys()), list(sorted_options.values()), color=colors_bar, edgecolor='white')

    # Add percentage labels
    for bar, (opt, count) in zip(bars, sorted_options.items()):
        pct = (count / total_seekers) * 100
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{count} ({pct:.0f}%)', va='center', fontsize=10)

    ax.set_xlabel('Number of Students', fontsize=12)
    ax.set_title(f'Dietary Options Students Actively Seek Out\n(Q10, n={total_seekers} respondents who seek options)',
                 fontsize=14, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No dietary options data available', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('outputs/fig46_dietary_options_sought.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/fig46_dietary_options_sought.png")

print("\n" + "=" * 60)
print("DIETARY VISUALIZATION GENERATION COMPLETE")
print("=" * 60)
print("\nGenerated figures:")
print("  - fig41_dietary_importance.png")
print("  - fig42_dietary_distribution.png")
print("  - fig43_dietary_segments.png")
print("  - fig44_dietary_correlations.png")
print("  - fig45_dietary_vs_quality.png")
print("  - fig46_dietary_options_sought.png")
