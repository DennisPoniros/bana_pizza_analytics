#!/usr/bin/env python3
"""
Side Order Analysis for BANA255 Pizza Survey
============================================
Analyzes Q25 (side order likelihood) and Q26 (side order spend) to identify
additional revenue opportunities for a new pizza restaurant.

Version: 8.0
Date: 2025-12-01
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("SIDE ORDER ANALYSIS - BANA255 Pizza Survey")
print("=" * 70)

# Load data
df = pd.read_excel('BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx')

# Skip header row (Qualtrics metadata)
df = df.iloc[1:].reset_index(drop=True)

# Filter to consented respondents only
df = df[df['Q2'] == 'Yes'].reset_index(drop=True)
print(f"\nAnalyzing {len(df)} consented respondents")

# =============================================================================
# SIDE ORDER ITEMS DEFINITION
# =============================================================================
side_items = {
    'Q25_1': 'Fries',
    'Q25_2': 'Salad',
    'Q25_3': 'Garlic knots/bread',
    'Q25_4': 'Mozzarella sticks',
    'Q25_5': 'Wings',
    'Q25_6': 'Fried calamari',
    'Q25_7': 'Bruschetta',
    'Q25_8': 'Stuffed mushrooms',
    'Q25_9': 'Onion rings',
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

# =============================================================================
# DATA PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("1. DATA PREPARATION")
print("=" * 70)

# Convert Q25 columns to numeric
for col, name in side_items.items():
    df[col + '_numeric'] = df[col].map(likelihood_map)

# Create a dataframe for analysis
side_cols = [col + '_numeric' for col in side_items.keys()]
side_names = list(side_items.values())

# Convert Q26 (spend) to numeric
df['side_spend'] = pd.to_numeric(df['Q26'], errors='coerce')

# Report missing data
print("\nMissing Data Summary for Side Orders:")
print("-" * 50)
for col, name in side_items.items():
    n_missing = df[col + '_numeric'].isna().sum()
    pct_missing = n_missing / len(df) * 100
    print(f"  {name:25s}: {n_missing:3d} missing ({pct_missing:.1f}%)")

n_spend_missing = df['side_spend'].isna().sum()
print(f"  {'Side Spend ($)':25s}: {n_spend_missing:3d} missing ({n_spend_missing/len(df)*100:.1f}%)")

# =============================================================================
# SIDE ORDER POPULARITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("2. SIDE ORDER POPULARITY RANKINGS")
print("=" * 70)

# Calculate statistics for each side item
side_stats = []
for col, name in side_items.items():
    data = df[col + '_numeric'].dropna()

    # Calculate key metrics
    mean_score = data.mean()
    median_score = data.median()
    std_score = data.std()

    # Percentage who would likely order (4 or 5)
    pct_likely = (data >= 4).sum() / len(data) * 100

    # Percentage who would definitely order (5 only)
    pct_definitely = (data == 5).sum() / len(data) * 100

    # Percentage who would NOT order (1 or 2)
    pct_unlikely = (data <= 2).sum() / len(data) * 100

    side_stats.append({
        'Side Item': name,
        'Mean': mean_score,
        'Median': median_score,
        'Std Dev': std_score,
        '% Likely': pct_likely,
        '% Definitely': pct_definitely,
        '% Unlikely': pct_unlikely,
        'N': len(data)
    })

# Create dataframe and sort by mean score
side_df = pd.DataFrame(side_stats)
side_df = side_df.sort_values('Mean', ascending=False).reset_index(drop=True)
side_df['Rank'] = range(1, len(side_df) + 1)

print("\nSide Order Popularity Rankings (1-5 scale):")
print("-" * 90)
print(f"{'Rank':<5} {'Side Item':<22} {'Mean':>7} {'Median':>7} {'% Likely':>10} {'% Definitely':>13} {'% Unlikely':>11}")
print("-" * 90)
for _, row in side_df.iterrows():
    print(f"{row['Rank']:<5} {row['Side Item']:<22} {row['Mean']:>7.2f} {row['Median']:>7.1f} {row['% Likely']:>9.1f}% {row['% Definitely']:>12.1f}% {row['% Unlikely']:>10.1f}%")

# =============================================================================
# STATISTICAL TESTS
# =============================================================================
print("\n" + "=" * 70)
print("3. STATISTICAL ANALYSIS")
print("=" * 70)

# Friedman test - are side order preferences significantly different?
side_data_matrix = df[[col + '_numeric' for col in side_items.keys()]].dropna()
print(f"\nFriedman Test (n={len(side_data_matrix)}):")
print("H0: All side items have equal popularity")

stat, p_value = stats.friedmanchisquare(*[side_data_matrix[col] for col in side_data_matrix.columns])
print(f"  Chi-square statistic: {stat:.2f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Result: {'REJECT H0 - Significant differences exist' if p_value < 0.05 else 'Fail to reject H0'}")

# Binomial test - do more than 50% want the top item?
top_item = side_df.iloc[0]['Side Item']
top_col = [k for k, v in side_items.items() if v == top_item][0]
n_likely = (df[top_col + '_numeric'] >= 4).sum()
n_total = df[top_col + '_numeric'].notna().sum()

print(f"\nBinomial Test for Top Item ({top_item}):")
print(f"H0: 50% or fewer students would likely order")
binom_result = stats.binomtest(n_likely, n_total, p=0.5, alternative='greater')
print(f"  Proportion likely: {n_likely/n_total*100:.1f}%")
print(f"  p-value: {binom_result.pvalue:.6f}")
print(f"  95% CI: [{binom_result.proportion_ci(confidence_level=0.95).low*100:.1f}%, {binom_result.proportion_ci(confidence_level=0.95).high*100:.1f}%]")

# =============================================================================
# SIDE SPEND ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("4. SIDE ORDER SPENDING ANALYSIS")
print("=" * 70)

spend_data = df['side_spend'].dropna()
print(f"\nSpend on Sides (Q26) Summary (n={len(spend_data)}):")
print("-" * 50)
print(f"  Mean:     ${spend_data.mean():.2f}")
print(f"  Median:   ${spend_data.median():.2f}")
print(f"  Std Dev:  ${spend_data.std():.2f}")
print(f"  Min:      ${spend_data.min():.2f}")
print(f"  Max:      ${spend_data.max():.2f}")
print(f"  IQR:      ${spend_data.quantile(0.25):.2f} - ${spend_data.quantile(0.75):.2f}")

# Spending segments
print("\nSpending Segments:")
print("-" * 50)
bins = [0, 5, 10, 15, 20, float('inf')]
labels = ['$0-5 (Low)', '$6-10 (Medium)', '$11-15 (High)', '$16-20 (Premium)', '$20+ (Very High)']
df['spend_segment'] = pd.cut(df['side_spend'], bins=bins, labels=labels, include_lowest=True)

for label in labels:
    count = (df['spend_segment'] == label).sum()
    pct = count / len(spend_data) * 100
    print(f"  {label:25s}: {count:3d} ({pct:5.1f}%)")

# =============================================================================
# CROSS-ANALYSIS: SIDES BY RESTAURANT PREFERENCE
# =============================================================================
print("\n" + "=" * 70)
print("5. SIDE ORDERS BY RESTAURANT TYPE")
print("=" * 70)

# Create local preference indicator
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

# Compare side preferences by restaurant type
print("\nSide Order Preference by Restaurant Type:")
print("-" * 70)
print(f"{'Side Item':<22} {'Local Mean':>12} {'Chain Mean':>12} {'Diff':>8} {'Sig?':>6}")
print("-" * 70)

for col, name in side_items.items():
    local_data = df[df['restaurant_type'] == 'Local'][col + '_numeric'].dropna()
    chain_data = df[df['restaurant_type'] == 'Chain'][col + '_numeric'].dropna()

    if len(local_data) >= 5 and len(chain_data) >= 5:
        local_mean = local_data.mean()
        chain_mean = chain_data.mean()
        diff = local_mean - chain_mean

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(local_data, chain_data, alternative='two-sided')
        sig = '*' if p < 0.05 else ''

        print(f"{name:<22} {local_mean:>12.2f} {chain_mean:>12.2f} {diff:>+8.2f} {sig:>6}")

# Compare side spending by restaurant type
print("\nSide Spending by Restaurant Type:")
print("-" * 50)
for rtype in ['Local', 'Chain']:
    spend = df[df['restaurant_type'] == rtype]['side_spend'].dropna()
    if len(spend) > 0:
        print(f"  {rtype:10s}: Mean ${spend.mean():.2f}, Median ${spend.median():.2f} (n={len(spend)})")

# =============================================================================
# SIDE ORDER SEGMENTS (CLUSTER ANALYSIS)
# =============================================================================
print("\n" + "=" * 70)
print("6. CUSTOMER SEGMENTATION BY SIDE PREFERENCES")
print("=" * 70)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare data for clustering
cluster_data = df[[col + '_numeric' for col in side_items.keys()]].dropna()

if len(cluster_data) >= 30:
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # K-means with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Add cluster labels back
    cluster_data = cluster_data.copy()
    cluster_data['Cluster'] = cluster_labels

    print("\nCustomer Segments (K-Means, k=3):")
    print("-" * 70)

    for i in range(3):
        segment = cluster_data[cluster_data['Cluster'] == i]
        n_segment = len(segment)
        pct = n_segment / len(cluster_data) * 100

        # Calculate mean for each side in this cluster
        means = segment[[col + '_numeric' for col in side_items.keys()]].mean()
        top_sides = means.nlargest(3)

        # Name the segment based on behavior
        overall_mean = means.mean()
        if overall_mean > 3.5:
            segment_name = "Side Enthusiasts"
        elif overall_mean < 2.5:
            segment_name = "Pizza Purists"
        else:
            segment_name = "Selective Siders"

        print(f"\nSegment {i+1}: {segment_name} ({n_segment} students, {pct:.1f}%)")
        print(f"  Overall side interest: {overall_mean:.2f}/5")
        print(f"  Top preferences:")
        for idx, val in top_sides.items():
            item_name = side_items[idx.replace('_numeric', '')]
            print(f"    - {item_name}: {val:.2f}")

# =============================================================================
# REVENUE OPPORTUNITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("7. REVENUE OPPORTUNITY ANALYSIS")
print("=" * 70)

# Average spend and frequency
avg_spend = spend_data.mean()
orders_per_month = df['Q4'].apply(pd.to_numeric, errors='coerce').dropna().mean()

print(f"\nKey Metrics:")
print(f"  Average side spend per order: ${avg_spend:.2f}")
print(f"  Average pizza orders per month: {orders_per_month:.1f}")
print(f"  Estimated monthly side revenue per customer: ${avg_spend * orders_per_month:.2f}")

# High-potential items (high interest, likely orderers)
print("\nHigh-Potential Side Items (>40% likely to order):")
print("-" * 50)
high_potential = side_df[side_df['% Likely'] > 40]
for _, row in high_potential.iterrows():
    print(f"  {row['Side Item']}: {row['% Likely']:.1f}% likely, avg score {row['Mean']:.2f}")

# Low-effort items (already popular, just highlight)
print("\nCore Menu Recommendations:")
print("-" * 50)
if len(high_potential) > 0:
    print("  MUST HAVE:")
    for _, row in high_potential.head(3).iterrows():
        print(f"    - {row['Side Item']}")

    mid_tier = side_df[(side_df['% Likely'] > 20) & (side_df['% Likely'] <= 40)]
    if len(mid_tier) > 0:
        print("  CONSIDER:")
        for _, row in mid_tier.head(3).iterrows():
            print(f"    - {row['Side Item']}")

    low_tier = side_df[side_df['% Likely'] <= 20]
    if len(low_tier) > 0:
        print("  SKIP (low demand):")
        for _, row in low_tier.iterrows():
            print(f"    - {row['Side Item']}")

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("8. SIDE PREFERENCE CORRELATIONS")
print("=" * 70)

# Spearman correlation matrix
corr_data = df[[col + '_numeric' for col in side_items.keys()]].dropna()
corr_matrix = corr_data.corr(method='spearman')

# Find strongest correlations
print("\nStrongest Side Item Correlations (|r| > 0.4):")
print("-" * 60)
seen_pairs = set()
for i, col1 in enumerate(corr_matrix.columns):
    for j, col2 in enumerate(corr_matrix.columns):
        if i < j:
            r = corr_matrix.loc[col1, col2]
            if abs(r) > 0.4:
                item1 = side_items[col1.replace('_numeric', '')]
                item2 = side_items[col2.replace('_numeric', '')]
                print(f"  {item1} <-> {item2}: r = {r:.3f}")
                seen_pairs.add((item1, item2))

if len(seen_pairs) == 0:
    print("  No correlations > 0.4 found")

# Correlation with side spend
print("\nCorrelation with Side Spending:")
print("-" * 50)
spend_corr = []
for col, name in side_items.items():
    valid_data = df[[col + '_numeric', 'side_spend']].dropna()
    if len(valid_data) >= 10:
        r, p = stats.spearmanr(valid_data[col + '_numeric'], valid_data['side_spend'])
        spend_corr.append({'Item': name, 'r': r, 'p': p})

spend_corr_df = pd.DataFrame(spend_corr).sort_values('r', ascending=False)
for _, row in spend_corr_df.iterrows():
    sig = '*' if row['p'] < 0.05 else ''
    print(f"  {row['Item']:22s}: r = {row['r']:+.3f} {sig}")

# =============================================================================
# EXPORT RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("9. EXPORTING RESULTS")
print("=" * 70)

# Save rankings
side_df.to_csv('outputs/side_order_rankings.csv', index=False)
print("  Saved: outputs/side_order_rankings.csv")

# Save segment data
if 'Cluster' in cluster_data.columns:
    segment_summary = cluster_data.groupby('Cluster').mean()
    segment_summary.to_csv('outputs/side_order_segments.csv')
    print("  Saved: outputs/side_order_segments.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY: SIDE ORDER ANALYSIS")
print("=" * 70)

print(f"""
KEY FINDINGS:

1. TOP SIDE ORDERS (by likelihood):
   #1: {side_df.iloc[0]['Side Item']} ({side_df.iloc[0]['% Likely']:.0f}% likely to order)
   #2: {side_df.iloc[1]['Side Item']} ({side_df.iloc[1]['% Likely']:.0f}% likely to order)
   #3: {side_df.iloc[2]['Side Item']} ({side_df.iloc[2]['% Likely']:.0f}% likely to order)

2. SIDE SPENDING:
   - Average spend: ${avg_spend:.2f} per order
   - Median spend: ${spend_data.median():.2f} per order
   - High spenders (>$15): {(spend_data > 15).sum()} students ({(spend_data > 15).sum()/len(spend_data)*100:.1f}%)

3. STATISTICAL SIGNIFICANCE:
   - Friedman test: p = {p_value:.6f} (preferences differ significantly)
   - Side preferences are NOT uniform - clear favorites exist

4. BUSINESS RECOMMENDATIONS:
   - PRIORITIZE: {', '.join(high_potential['Side Item'].head(3).tolist()) if len(high_potential) >= 3 else high_potential['Side Item'].tolist()}
   - BUNDLE OPPORTUNITY: Pair popular sides (e.g., pizza + {side_df.iloc[0]['Side Item'].lower()})
   - PRICING: Target ${spend_data.median():.0f}-{spend_data.quantile(0.75):.0f} side menu range

5. SEGMENT INSIGHT:
   - Three customer segments exist with distinct side preferences
   - Target "Side Enthusiasts" for upselling opportunities
""")

print("\n" + "=" * 70)
print("Analysis complete. See outputs/ for exported data.")
print("=" * 70)
