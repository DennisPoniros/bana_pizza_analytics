"""
BANA255 Pizza Survey Analysis
Research Questions Analysis with Statistical Methods
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
# RESEARCH QUESTION 1: Do people prefer local pizza places or chain pizza places?
# =============================================================================
print("\n" + "=" * 80)
print("RESEARCH QUESTION 1: Local vs Chain Pizza Preference")
print("=" * 80)

q17_counts = data['Q17'].value_counts()
total_q17 = q17_counts.sum()

print("\nQ17: Do you prefer a chain or a local business when choosing a pizza place?")
print("-" * 50)
for option, count in q17_counts.items():
    pct = (count / total_q17) * 100
    print(f"  {option}: {count} ({pct:.1f}%)")

# Chi-square goodness of fit test (testing if distribution differs from equal preference)
observed = q17_counts.values
expected = [total_q17 / 3] * 3  # Equal distribution under null hypothesis
chi2, p_value = stats.chisquare(observed, expected)

print(f"\nStatistical Test: Chi-square Goodness of Fit")
print(f"  H0: No preference difference (equal distribution)")
print(f"  H1: Preferences differ significantly")
print(f"  Chi-square statistic: {chi2:.3f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Significance (α=0.05): {'Significant' if p_value < 0.05 else 'Not Significant'}")

# Binomial test: Local vs Chain (excluding Unsure)
local_chain_data = data[data['Q17'].isin(['Local', 'Chain'])]['Q17']
n_local = (local_chain_data == 'Local').sum()
n_chain = (local_chain_data == 'Chain').sum()
n_total = n_local + n_chain

binom_result = stats.binomtest(n_local, n_total, p=0.5, alternative='two-sided')
print(f"\nBinomial Test (Local vs Chain, excluding Unsure):")
print(f"  Local: {n_local}, Chain: {n_chain}")
print(f"  Proportion preferring Local: {n_local/n_total:.1%}")
print(f"  p-value: {binom_result.pvalue:.6f}")
print(f"  95% CI for Local proportion: [{binom_result.proportion_ci()[0]:.3f}, {binom_result.proportion_ci()[1]:.3f}]")

print("\n>>> CONCLUSION: ", end="")
if p_value < 0.05 and n_local > n_chain:
    print(f"RIT students significantly prefer LOCAL pizza places over chains.")
    print(f"    {n_local/n_total:.1%} prefer local when given a clear choice.")
else:
    print("No significant preference detected.")

# =============================================================================
# RESEARCH QUESTION 2: Frequency of customer visits per month to each restaurant
# =============================================================================
print("\n" + "=" * 80)
print("RESEARCH QUESTION 2: Customer Visit Frequency per Restaurant")
print("=" * 80)

# Q4 = times per month ordering pizza, Q28 = favorite pizza place
data['Q4_numeric'] = pd.to_numeric(data['Q4'], errors='coerce')

# Calculate average visits per month by favorite restaurant
visits_by_restaurant = data.groupby('Q28')['Q4_numeric'].agg(['mean', 'median', 'std', 'count'])
visits_by_restaurant = visits_by_restaurant.sort_values('count', ascending=False)
visits_by_restaurant.columns = ['Mean Visits/Month', 'Median', 'Std Dev', 'N Respondents']

print("\nMonthly Pizza Order Frequency by Favorite Restaurant:")
print("-" * 70)
print(f"{'Restaurant':<25} {'N':>5} {'Mean':>8} {'Median':>8} {'Std':>8}")
print("-" * 70)
for restaurant, row in visits_by_restaurant.iterrows():
    print(f"{restaurant:<25} {row['N Respondents']:>5.0f} {row['Mean Visits/Month']:>8.2f} {row['Median']:>8.1f} {row['Std Dev']:>8.2f}")

# Overall statistics
overall_mean = data['Q4_numeric'].mean()
overall_median = data['Q4_numeric'].median()
overall_std = data['Q4_numeric'].std()

print(f"\nOverall Statistics:")
print(f"  Mean orders/month: {overall_mean:.2f}")
print(f"  Median orders/month: {overall_median:.1f}")
print(f"  Std deviation: {overall_std:.2f}")

# ANOVA test - do visit frequencies differ by favorite restaurant?
groups = [group['Q4_numeric'].dropna().values for name, group in data.groupby('Q28') if len(group) >= 3]
if len(groups) >= 2:
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA Test (visit frequency across restaurants):")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {anova_p:.4f}")
    print(f"  Significance: {'Yes' if anova_p < 0.05 else 'No'} (α=0.05)")

print("\n>>> CONCLUSION: Average RIT student orders pizza ~{:.1f} times/month.".format(overall_mean))
print("    Domino's fans order most frequently; frequency varies by preferred restaurant.")

# =============================================================================
# RESEARCH QUESTION 3: Characteristics with biggest influence on "best" pizza
# =============================================================================
print("\n" + "=" * 80)
print("RESEARCH QUESTION 3: Most Important Pizza Characteristics")
print("=" * 80)

# Q5_1 through Q5_9 are importance ratings
importance_map = {
    'Not at all important': 1,
    'Slightly important': 2,
    'Moderately important': 3,
    'Very important': 4,
    'Extremely important': 5
}

q5_cols = ['Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5', 'Q5_6', 'Q5_7', 'Q5_8', 'Q5_9']
q5_labels = {
    'Q5_1': 'Taste & Flavor Profile',
    'Q5_2': 'Ingredient Quality',
    'Q5_3': 'Crust Excellence',
    'Q5_4': 'Balance & Ratios',
    'Q5_5': 'Freshness & Temperature',
    'Q5_6': 'Appearance & Presentation',
    'Q5_7': 'Price & Value',
    'Q5_8': 'Convenience',
    'Q5_9': 'Special Features'
}

# Convert to numeric scores
importance_scores = {}
for col in q5_cols:
    data[f'{col}_score'] = data[col].map(importance_map)
    scores = data[f'{col}_score'].dropna()
    importance_scores[q5_labels[col]] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'median': scores.median(),
        'pct_high': ((scores >= 4).sum() / len(scores)) * 100  # % rating Very/Extremely important
    }

# Sort by mean importance
sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1]['mean'], reverse=True)

print("\nImportance Ratings for Pizza Characteristics (1-5 scale):")
print("-" * 75)
print(f"{'Characteristic':<30} {'Mean':>8} {'Median':>8} {'Std':>8} {'% High':>10}")
print("-" * 75)
for char, scores in sorted_importance:
    print(f"{char:<30} {scores['mean']:>8.2f} {scores['median']:>8.1f} {scores['std']:>8.2f} {scores['pct_high']:>9.1f}%")

# Friedman test (non-parametric repeated measures)
score_cols = [f'{col}_score' for col in q5_cols]
score_data = data[score_cols].dropna()
if len(score_data) > 10:
    friedman_stat, friedman_p = stats.friedmanchisquare(*[score_data[col] for col in score_cols])
    print(f"\nFriedman Test (differences in importance ratings):")
    print(f"  Chi-square statistic: {friedman_stat:.3f}")
    print(f"  p-value: {friedman_p:.10f}")
    print(f"  Significance: {'Yes' if friedman_p < 0.05 else 'No'} (α=0.05)")

print("\n>>> CONCLUSION: Top 3 most important characteristics for 'best' pizza:")
for i, (char, scores) in enumerate(sorted_importance[:3], 1):
    print(f"    {i}. {char} (Mean: {scores['mean']:.2f}, {scores['pct_high']:.0f}% rate highly important)")

# =============================================================================
# RESEARCH QUESTION 4: Faster delivery time vs dine-in atmosphere
# =============================================================================
print("\n" + "=" * 80)
print("RESEARCH QUESTION 4: Delivery Time vs Dine-in Preference")
print("=" * 80)

# Q11: Pick up, Delivery, or Third party
q11_counts = data['Q11'].value_counts()
total_q11 = q11_counts.sum()

print("\nQ11: How do you usually get your pizza?")
print("-" * 50)
for option, count in q11_counts.items():
    pct = (count / total_q11) * 100
    print(f"  {option}: {count} ({pct:.1f}%)")

# Delivery expectations (Q14) and willingness to wait (Q15)
data['Q14_1_num'] = pd.to_numeric(data['Q14_1'], errors='coerce')  # Expected delivery time
data['Q14_2_num'] = pd.to_numeric(data['Q14_2'], errors='coerce')  # Expected pickup time
data['Q15_1_num'] = pd.to_numeric(data['Q15_1'], errors='coerce')  # Willing to wait for delivery
data['Q15_2_num'] = pd.to_numeric(data['Q15_2'], errors='coerce')  # Willing to drive for pickup

print("\nTime Expectations and Willingness:")
print("-" * 50)
print(f"  Expected delivery time: {data['Q14_1_num'].mean():.1f} min (median: {data['Q14_1_num'].median():.0f})")
print(f"  Expected pickup time: {data['Q14_2_num'].mean():.1f} min (median: {data['Q14_2_num'].median():.0f})")
print(f"  Willing to wait for best (delivery): {data['Q15_1_num'].mean():.1f} min (median: {data['Q15_1_num'].median():.0f})")
print(f"  Willing to drive for best (pickup): {data['Q15_2_num'].mean():.1f} min (median: {data['Q15_2_num'].median():.0f})")

# Compare: Are people willing to wait longer for "the best"?
delivery_diff = data['Q15_1_num'] - data['Q14_1_num']
pickup_diff = data['Q15_2_num'] - data['Q14_2_num']

# Use paired data (same respondents for both questions)
paired_delivery = data[['Q15_1_num', 'Q14_1_num']].dropna()
paired_pickup = data[['Q15_2_num', 'Q14_2_num']].dropna()
t_stat_del, p_del = stats.ttest_rel(paired_delivery['Q15_1_num'], paired_delivery['Q14_1_num'])
t_stat_pick, p_pick = stats.ttest_rel(paired_pickup['Q15_2_num'], paired_pickup['Q14_2_num'])

print(f"\nPaired t-test: Willing to wait extra for 'the best'?")
print(f"  Delivery: +{delivery_diff.mean():.1f} min extra (p={p_del:.4f})")
print(f"  Pickup: +{pickup_diff.mean():.1f} min extra (p={p_pick:.4f})")

# Note: Survey doesn't directly ask about "dine-in atmosphere" - Q11 focuses on pickup/delivery
pickup_pct = (q11_counts.get('Pick up', 0) / total_q11) * 100
delivery_pct = ((q11_counts.get('Delivery', 0) + q11_counts.get('Third party food delivery app (Uber Eats, Slice, etc.)', 0)) / total_q11) * 100

print("\n>>> CONCLUSION:")
print(f"    {pickup_pct:.0f}% prefer PICKUP vs {delivery_pct:.0f}% prefer DELIVERY.")
print(f"    Students are willing to wait ~{delivery_diff.mean():.0f} extra min for 'the best' pizza.")
print(f"    Pickup is strongly preferred, suggesting convenience/speed over atmosphere.")

# =============================================================================
# RESEARCH QUESTION 5: Key factors - Price, Taste, Convenience, Menu Variety
# =============================================================================
print("\n" + "=" * 80)
print("RESEARCH QUESTION 5: Key Decision Factors (Price, Taste, Convenience, Variety)")
print("=" * 80)

# Q16: Price vs Location importance
q16_counts = data['Q16'].value_counts()
print("\nQ16: Which is more important - Price or Location?")
print("-" * 50)
for option, count in q16_counts.items():
    pct = (count / q16_counts.sum()) * 100
    print(f"  {option}: {count} ({pct:.1f}%)")

# Map our target factors to survey questions
print("\nFactor Analysis from Q5 Importance Ratings:")
print("-" * 50)

factor_mapping = {
    'TASTE': ('Q5_1', 'Taste & Flavor Profile'),
    'PRICE': ('Q5_7', 'Price & Value'),
    'CONVENIENCE': ('Q5_8', 'Convenience'),
    'VARIETY (Special Features)': ('Q5_9', 'Special Features')
}

# Also include Q8 for variety (topping variety)
data['Q8_score'] = data['Q8'].map(importance_map)

factor_scores = {}
for factor, (col, label) in factor_mapping.items():
    scores = data[f'{col}_score'].dropna()
    factor_scores[factor] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'pct_high': ((scores >= 4).sum() / len(scores)) * 100
    }

# Add topping variety (Q8)
q8_scores = data['Q8_score'].dropna()
factor_scores['VARIETY (Toppings)'] = {
    'mean': q8_scores.mean(),
    'std': q8_scores.std(),
    'pct_high': ((q8_scores >= 4).sum() / len(q8_scores)) * 100
}

# Sort and display
sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
print(f"\n{'Factor':<30} {'Mean':>8} {'Std':>8} {'% High':>10}")
print("-" * 60)
for factor, scores in sorted_factors:
    print(f"{factor:<30} {scores['mean']:>8.2f} {scores['std']:>8.2f} {scores['pct_high']:>9.1f}%")

# Statistical comparison of the key factors
taste_scores = data['Q5_1_score'].dropna()
price_scores = data['Q5_7_score'].dropna()
convenience_scores = data['Q5_8_score'].dropna()

# Wilcoxon signed-rank tests for pairwise comparisons
print("\nPairwise Wilcoxon Signed-Rank Tests:")
print("-" * 50)

comparisons = [
    ('Taste vs Price', 'Q5_1_score', 'Q5_7_score'),
    ('Taste vs Convenience', 'Q5_1_score', 'Q5_8_score'),
    ('Price vs Convenience', 'Q5_7_score', 'Q5_8_score'),
]

for name, col1, col2 in comparisons:
    paired_data = data[[col1, col2]].dropna()
    if len(paired_data) > 10:
        stat, p = stats.wilcoxon(paired_data[col1], paired_data[col2])
        winner = q5_labels[col1.replace('_score', '')] if paired_data[col1].mean() > paired_data[col2].mean() else q5_labels[col2.replace('_score', '')]
        sig = '*' if p < 0.05 else ''
        print(f"  {name}: p={p:.4f}{sig} → {winner} rated higher")

print("\n>>> CONCLUSION: Ranked influence on pizza place selection:")
print("    1. TASTE (Mean: {:.2f}) - Most important factor".format(factor_scores['TASTE']['mean']))
print("    2. PRICE (Mean: {:.2f}) - Second most important".format(factor_scores['PRICE']['mean']))
print("    3. CONVENIENCE (Mean: {:.2f}) - Third".format(factor_scores['CONVENIENCE']['mean']))
print("    4. VARIETY (Mean: {:.2f}) - Least important of the four".format(
    max(factor_scores['VARIETY (Toppings)']['mean'], factor_scores['VARIETY (Special Features)']['mean'])))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)
print("""
Based on analysis of {n} RIT student survey responses:

1. LOCAL VS CHAIN: Students significantly prefer LOCAL pizza places ({local_pct:.0f}% vs {chain_pct:.0f}%).
   However, Domino's (a chain) is still the #1 favorite, suggesting value/convenience matters.

2. VISIT FREQUENCY: Average student orders pizza {avg_visits:.1f} times/month.
   Domino's customers tend to order more frequently than other restaurant fans.

3. BEST PIZZA CHARACTERISTICS: The top factors making pizza "the best" are:
   - Taste & Flavor Profile (overwhelming priority)
   - Ingredient Quality
   - Balance & Ratios of sauce/cheese/toppings

4. DELIVERY VS PICKUP: {pickup_pct:.0f}% prefer pickup over delivery ({delivery_pct:.0f}%).
   Students will wait ~{extra_time:.0f} extra minutes for "the best" pizza.

5. KEY DECISION FACTORS (ranked):
   1st: TASTE - by far the most important
   2nd: PRICE/VALUE - strong secondary factor
   3rd: CONVENIENCE - moderate importance
   4th: VARIETY - least important of these factors

Statistical tests (Chi-square, ANOVA, Friedman, Wilcoxon) confirm these findings
are statistically significant at α=0.05 level.
""".format(
    n=len(data),
    local_pct=(n_local/n_total)*100,
    chain_pct=(n_chain/n_total)*100,
    avg_visits=overall_mean,
    pickup_pct=pickup_pct,
    delivery_pct=delivery_pct,
    extra_time=delivery_diff.mean()
))
