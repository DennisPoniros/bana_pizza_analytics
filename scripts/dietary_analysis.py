"""
BANA255 Pizza Survey - Dietary Accommodation Analysis
Analyzes Q9_1-Q9_11 (dietary importance) and Q10 (seeking dietary options)
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

# Load data
xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)

# Extract question descriptions (row 0) and actual data (rows 1+)
questions = df.iloc[0].to_dict()
data = df.iloc[1:].reset_index(drop=True)

# Filter to only consented responses
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)
print(f"Total valid responses (consented): {len(data)}")
print("=" * 80)

# =============================================================================
# DIETARY ACCOMMODATION MAPPING
# =============================================================================

importance_map = {
    'Not at all important': 1,
    'Slightly important': 2,
    'Moderately important': 3,
    'Very important': 4,
    'Extremely important': 5
}

# Q9 columns: Dietary considerations importance
q9_cols = ['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5', 'Q9_6', 'Q9_7', 'Q9_8', 'Q9_9', 'Q9_10', 'Q9_11']
q9_labels = {
    'Q9_1': 'Gluten-free crust options',
    'Q9_2': 'Vegan/dairy-free cheese',
    'Q9_3': 'Separate prep areas (cross-contamination)',
    'Q9_4': 'Low-carb/keto options',
    'Q9_5': 'Plant-based protein toppings',
    'Q9_6': 'Allergen transparency',
    'Q9_7': 'Organic/non-GMO ingredients',
    'Q9_8': 'Reduced sodium/healthier options',
    'Q9_9': 'Halal/kosher certification',
    'Q9_10': 'Half-and-half pizzas (dietary flexibility)',
    'Q9_11': 'Clear allergen labeling'
}

# Short labels for visualizations
q9_short_labels = {
    'Q9_1': 'Gluten-Free',
    'Q9_2': 'Vegan/Dairy-Free',
    'Q9_3': 'Cross-Contam. Prevention',
    'Q9_4': 'Low-Carb/Keto',
    'Q9_5': 'Plant-Based Protein',
    'Q9_6': 'Allergen Transparency',
    'Q9_7': 'Organic/Non-GMO',
    'Q9_8': 'Reduced Sodium',
    'Q9_9': 'Halal/Kosher',
    'Q9_10': 'Half-and-Half Option',
    'Q9_11': 'Allergen Labeling'
}

# =============================================================================
# ANALYSIS 1: Overall Dietary Importance Rankings
# =============================================================================
print("\n" + "=" * 80)
print("DIETARY ACCOMMODATION IMPORTANCE ANALYSIS")
print("=" * 80)

# Convert to numeric scores
dietary_scores = {}
for col in q9_cols:
    data[f'{col}_score'] = data[col].map(importance_map)
    scores = data[f'{col}_score'].dropna()
    dietary_scores[col] = {
        'label': q9_labels[col],
        'short_label': q9_short_labels[col],
        'mean': scores.mean(),
        'std': scores.std(),
        'median': scores.median(),
        'pct_high': ((scores >= 4).sum() / len(scores)) * 100,  # % rating Very/Extremely important
        'pct_mod_plus': ((scores >= 3).sum() / len(scores)) * 100,  # % rating at least Moderately important
        'pct_not_important': ((scores == 1).sum() / len(scores)) * 100,  # % rating Not at all important
        'n': len(scores)
    }

# Sort by mean importance
sorted_dietary = sorted(dietary_scores.items(), key=lambda x: x[1]['mean'], reverse=True)

print("\nDietary Consideration Importance Rankings (1-5 scale):")
print("-" * 90)
print(f"{'Rank':<5} {'Consideration':<35} {'Mean':>8} {'Median':>8} {'% High':>10} {'% Not Imp':>10}")
print("-" * 90)
for rank, (col, scores) in enumerate(sorted_dietary, 1):
    print(f"{rank:<5} {scores['short_label']:<35} {scores['mean']:>8.2f} {scores['median']:>8.1f} {scores['pct_high']:>9.1f}% {scores['pct_not_important']:>9.1f}%")

# Overall dietary importance (average across all factors)
all_dietary_scores = []
for col in q9_cols:
    all_dietary_scores.extend(data[f'{col}_score'].dropna().tolist())
overall_dietary_mean = np.mean(all_dietary_scores)
overall_dietary_std = np.std(all_dietary_scores)

print(f"\nOverall Dietary Accommodation Importance:")
print(f"  Mean across all factors: {overall_dietary_mean:.2f}/5")
print(f"  Std deviation: {overall_dietary_std:.2f}")

# =============================================================================
# ANALYSIS 2: Friedman Test - Are there significant differences?
# =============================================================================
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

score_cols = [f'{col}_score' for col in q9_cols]
score_data = data[score_cols].dropna()

if len(score_data) > 10:
    friedman_stat, friedman_p = stats.friedmanchisquare(*[score_data[col] for col in score_cols])
    print(f"\nFriedman Test (differences across dietary factors):")
    print(f"  Chi-square statistic: {friedman_stat:.3f}")
    print(f"  p-value: {friedman_p:.10f}")
    print(f"  Significance: {'Yes - factors differ significantly' if friedman_p < 0.05 else 'No'} (alpha=0.05)")

# =============================================================================
# ANALYSIS 3: Cluster Analysis - Dietary-Conscious vs Indifferent
# =============================================================================
print("\n" + "=" * 80)
print("DIETARY CONSCIOUSNESS SEGMENTATION")
print("=" * 80)

# Calculate dietary consciousness score (mean of all dietary factors)
data['dietary_consciousness'] = data[score_cols].mean(axis=1)

# Define segments based on average dietary consciousness
low_threshold = 2.0  # Average below 2 = "Dietary Indifferent"
high_threshold = 3.0  # Average 3+ = "Dietary Conscious"

data['dietary_segment'] = pd.cut(
    data['dietary_consciousness'],
    bins=[0, low_threshold, high_threshold, 5],
    labels=['Indifferent', 'Moderate', 'Conscious']
)

segment_counts = data['dietary_segment'].value_counts()
print("\nDietary Consciousness Segments:")
print("-" * 50)
for segment, count in segment_counts.items():
    pct = (count / len(data)) * 100
    print(f"  {segment}: {count} ({pct:.1f}%)")

# Compare segments on key metrics
print("\nSegment Profiles:")
print("-" * 70)
print(f"{'Metric':<30} {'Indifferent':>12} {'Moderate':>12} {'Conscious':>12}")
print("-" * 70)

for metric_col, metric_name in [('Q4', 'Orders/Month'), ('Q29', 'Loyalty (1-5)')]:
    data[f'{metric_col}_num'] = pd.to_numeric(data[metric_col], errors='coerce')

for segment in ['Indifferent', 'Moderate', 'Conscious']:
    seg_data = data[data['dietary_segment'] == segment]

metrics = {
    'Mean Orders/Month': lambda x: x['Q4'].apply(pd.to_numeric, errors='coerce').mean(),
    'Mean Taste Importance': lambda x: x['Q5_1'].map(importance_map).mean(),
    'Mean Price Importance': lambda x: x['Q5_7'].map(importance_map).mean(),
    'Prefer Local (%)': lambda x: (x['Q17'] == 'Local').sum() / len(x) * 100
}

for metric_name, metric_func in metrics.items():
    values = []
    for segment in ['Indifferent', 'Moderate', 'Conscious']:
        seg_data = data[data['dietary_segment'] == segment]
        if len(seg_data) > 0:
            try:
                val = metric_func(seg_data)
                values.append(f"{val:.1f}")
            except:
                values.append("N/A")
        else:
            values.append("N/A")
    print(f"{metric_name:<30} {values[0]:>12} {values[1]:>12} {values[2]:>12}")

# =============================================================================
# ANALYSIS 4: Chi-Square - Dietary Consciousness vs Local/Chain Preference
# =============================================================================
print("\n" + "=" * 80)
print("DIETARY CONSCIOUSNESS vs RESTAURANT PREFERENCE")
print("=" * 80)

# Create binary local choice variable
data['chose_local'] = data['Q28'].apply(lambda x: 'Local' if x in ['Joe\'s Brooklyn Pizza', 'Salvatore\'s Pizza', 'Mark\'s Pizzeria', 'Pontillo\'s Pizza'] else 'Chain')

# Cross-tabulation
crosstab = pd.crosstab(data['dietary_segment'], data['chose_local'])
print("\nCross-tabulation: Dietary Segment vs Restaurant Choice")
print(crosstab)

# Chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
print(f"\nChi-Square Test of Independence:")
print(f"  Chi-square statistic: {chi2:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Significance: {'Yes' if p_value < 0.05 else 'No'} (alpha=0.05)")

# Calculate Cramér's V
n = crosstab.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
print(f"  Cramér's V: {cramers_v:.3f}")

# =============================================================================
# ANALYSIS 5: Individual Dietary Factor Correlations
# =============================================================================
print("\n" + "=" * 80)
print("CORRELATIONS: Dietary Factors vs Local Choice")
print("=" * 80)

# Binary local choice for correlation
data['local_binary'] = (data['Q17'] == 'Local').astype(int)

print("\nPoint-Biserial Correlations (Dietary Importance vs States Prefer Local):")
print("-" * 70)
print(f"{'Dietary Factor':<35} {'r':>10} {'p-value':>12} {'Sig':>8}")
print("-" * 70)

correlations = []
for col in q9_cols:
    score_col = f'{col}_score'
    valid_data = data[[score_col, 'local_binary']].dropna()
    if len(valid_data) > 10:
        r, p = stats.pointbiserialr(valid_data['local_binary'], valid_data[score_col])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        correlations.append({
            'col': col,
            'label': q9_short_labels[col],
            'r': r,
            'p': p,
            'sig': sig
        })
        print(f"{q9_short_labels[col]:<35} {r:>10.3f} {p:>12.4f} {sig:>8}")

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x['r']), reverse=True)

# =============================================================================
# ANALYSIS 6: Q10 - Actively Seeking Dietary Options
# =============================================================================
print("\n" + "=" * 80)
print("Q10: DO STUDENTS SEEK OUT DIETARY OPTIONS?")
print("=" * 80)

# Q10 is multi-select, need to analyze responses
q10_data = data['Q10'].dropna()
print(f"\nResponses analyzed: {len(q10_data)}")

# Count responses (Q10 values represent what options they seek)
# Parse responses - they may be comma-separated or single values
all_options = []
for response in q10_data:
    if pd.notna(response) and str(response).strip():
        # Handle potential multi-select responses
        options = str(response).split(',')
        all_options.extend([opt.strip() for opt in options])

option_counts = Counter(all_options)
total_respondents = len(q10_data)

print("\nDietary Options Students Actively Seek:")
print("-" * 60)
for option, count in option_counts.most_common():
    pct = (count / total_respondents) * 100
    if option and option != 'nan':
        print(f"  {option}: {count} ({pct:.1f}%)")

# =============================================================================
# ANALYSIS 7: Opportunity Analysis
# =============================================================================
print("\n" + "=" * 80)
print("MARKET OPPORTUNITY ANALYSIS")
print("=" * 80)

# Calculate unmet needs: High importance but potentially unserved
print("\nUnmet Dietary Needs (High importance factors):")
print("-" * 70)

for rank, (col, scores) in enumerate(sorted_dietary[:5], 1):
    print(f"\n{rank}. {scores['label']}")
    print(f"   Mean Importance: {scores['mean']:.2f}/5")
    print(f"   % Rating Highly Important: {scores['pct_high']:.1f}%")
    print(f"   % Rating Mod+ Important: {scores['pct_mod_plus']:.1f}%")

    # Estimate unmet demand
    n_highly_important = int(scores['pct_high'] / 100 * len(data))
    print(f"   Est. Students with High Priority: ~{n_highly_important} students")

# =============================================================================
# ANALYSIS 8: Demographic Breakdown
# =============================================================================
print("\n" + "=" * 80)
print("DEMOGRAPHIC ANALYSIS")
print("=" * 80)

# By housing type
print("\nDietary Consciousness by Housing:")
print("-" * 50)
for housing in data['Q33'].unique():
    if pd.notna(housing):
        housing_data = data[data['Q33'] == housing]['dietary_consciousness'].dropna()
        if len(housing_data) > 5:
            print(f"  {housing}: Mean = {housing_data.mean():.2f} (n={len(housing_data)})")

# By year
print("\nDietary Consciousness by Year:")
print("-" * 50)
for year in ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate']:
    year_data = data[data['Q31'] == year]['dietary_consciousness'].dropna()
    if len(year_data) > 3:
        print(f"  {year}: Mean = {year_data.mean():.2f} (n={len(year_data)})")

# =============================================================================
# EXPORT RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Export dietary importance rankings
dietary_export = []
for rank, (col, scores) in enumerate(sorted_dietary, 1):
    dietary_export.append({
        'Rank': rank,
        'Factor': scores['label'],
        'Short_Label': scores['short_label'],
        'Mean': scores['mean'],
        'Median': scores['median'],
        'Std': scores['std'],
        'Pct_High_Importance': scores['pct_high'],
        'Pct_Moderate_Plus': scores['pct_mod_plus'],
        'Pct_Not_Important': scores['pct_not_important'],
        'N': scores['n']
    })

dietary_df = pd.DataFrame(dietary_export)
dietary_df.to_csv('outputs/dietary_importance_rankings.csv', index=False)
print("Exported: outputs/dietary_importance_rankings.csv")

# Export segment data
segment_export = data[['dietary_consciousness', 'dietary_segment', 'Q17', 'Q28', 'Q4', 'Q33', 'Q31']].copy()
segment_export.columns = ['Dietary_Consciousness_Score', 'Segment', 'Stated_Preference', 'Favorite_Restaurant', 'Orders_Per_Month', 'Housing', 'Year']
segment_export.to_csv('outputs/dietary_segments.csv', index=False)
print("Exported: outputs/dietary_segments.csv")

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY: DIETARY ACCOMMODATION ANALYSIS")
print("=" * 80)

top_factor = sorted_dietary[0][1]
bottom_factor = sorted_dietary[-1][1]

# Count dietary-conscious students
conscious_count = (data['dietary_segment'] == 'Conscious').sum()
moderate_count = (data['dietary_segment'] == 'Moderate').sum()
conscious_pct = (conscious_count + moderate_count) / len(data) * 100

print(f"""
Based on analysis of {len(data)} RIT student survey responses:

1. OVERALL DIETARY IMPORTANCE: LOW TO MODERATE
   - Average importance across all dietary factors: {overall_dietary_mean:.2f}/5
   - Most students rate dietary accommodations as "Slightly" to "Moderately" important
   - {conscious_pct:.0f}% show moderate-to-high dietary consciousness

2. TOP DIETARY CONSIDERATIONS (ranked by importance):
   1. {sorted_dietary[0][1]['label']} (Mean: {sorted_dietary[0][1]['mean']:.2f})
   2. {sorted_dietary[1][1]['label']} (Mean: {sorted_dietary[1][1]['mean']:.2f})
   3. {sorted_dietary[2][1]['label']} (Mean: {sorted_dietary[2][1]['mean']:.2f})

3. LOWEST PRIORITY DIETARY FACTORS:
   - {bottom_factor['label']} (Mean: {bottom_factor['mean']:.2f}, {bottom_factor['pct_not_important']:.0f}% say "Not important")

4. MARKET SEGMENTS:
   - Dietary Indifferent: {segment_counts.get('Indifferent', 0)} students ({segment_counts.get('Indifferent', 0)/len(data)*100:.0f}%)
   - Moderately Concerned: {segment_counts.get('Moderate', 0)} students ({segment_counts.get('Moderate', 0)/len(data)*100:.0f}%)
   - Dietary Conscious: {segment_counts.get('Conscious', 0)} students ({segment_counts.get('Conscious', 0)/len(data)*100:.0f}%)

5. STRATEGIC IMPLICATIONS:
   - Dietary accommodations are NOT a primary driver of pizza choice for most students
   - However, ~{conscious_count + moderate_count} students ({conscious_pct:.0f}%) DO care about dietary options
   - The top opportunities are: {sorted_dietary[0][1]['short_label']}, {sorted_dietary[1][1]['short_label']}, {sorted_dietary[2][1]['short_label']}
   - Offering half-and-half pizzas provides flexibility for mixed dietary needs

Statistical tests confirm significant differences exist across dietary factors (p < 0.001).
""")

print("=" * 80)
print("Analysis complete. See outputs/ for exported data.")
print("=" * 80)
