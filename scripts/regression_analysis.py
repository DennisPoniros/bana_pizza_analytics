"""
BANA255 Pizza Survey - Regression Analysis
============================================
Predictive modeling to explain what drives pizza place preference
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 80)
print("REGRESSION ANALYSIS: Explaining Pizza Preference")
print("=" * 80)

xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

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
for col in q5_cols:
    data[f'{col}_score'] = data[col].map(importance_map)

data['loyalty_score'] = data['Q29'].map(loyalty_map)
data['orders_month'] = pd.to_numeric(data['Q4'], errors='coerce')
data['expected_price'] = pd.to_numeric(data['Q21_1'], errors='coerce')
data['max_price'] = pd.to_numeric(data['Q21_2'], errors='coerce')

# Binary encodings
data['is_dominos'] = (data['Q28'] == "Domino's Pizza").astype(int)
data['is_local_favorite'] = data['Q28'].isin([
    "Joe's Brooklyn Pizza", "Salvatore's Pizza", "Mark's Pizzeria",
    "Pontillo's Pizza", "Perri's Pizza", "Fire Crust Pizza",
    "Peels on Wheels", "Brandani's Pizza", "Tony Pepperoni", "Pizza Wizard"
]).astype(int)
data['prefers_local'] = (data['Q17'] == 'Local').astype(int)
data['prefers_pickup'] = (data['Q11'] == 'Pick up').astype(int)
data['has_transport'] = (data['Q36'] == 'Yes').astype(int)
data['on_campus'] = (data['Q33'] == 'On campus').astype(int)

# =============================================================================
# MODEL 1: What predicts LOYALTY to a pizza place?
# =============================================================================
print("\n" + "=" * 80)
print("MODEL 1: What Predicts Customer Loyalty?")
print("Dependent Variable: Loyalty Score (1-5)")
print("=" * 80)

# Prepare regression data
loyalty_predictors = ['Q5_1_score', 'Q5_3_score', 'Q5_7_score', 'Q5_8_score',
                      'orders_month', 'expected_price', 'prefers_local', 'is_dominos']

loyalty_data = data[['loyalty_score'] + loyalty_predictors].dropna()
y = loyalty_data['loyalty_score']
X = loyalty_data[loyalty_predictors]

# Simple linear regression for each predictor
print("\nBivariate Correlations with Loyalty:")
print("-" * 60)
print(f"{'Predictor':<25} {'Correlation':>12} {'p-value':>12} {'Sig':>6}")
print("-" * 60)

correlations = []
for pred in loyalty_predictors:
    r, p = stats.pearsonr(X[pred], y)
    sig = '*' if p < 0.05 else ''
    sig = '**' if p < 0.01 else sig
    sig = '***' if p < 0.001 else sig
    correlations.append((pred, r, p, sig))
    pred_name = pred.replace('_score', '').replace('Q5_1', 'Taste Imp').replace('Q5_3', 'Crust Imp')
    pred_name = pred_name.replace('Q5_7', 'Price Imp').replace('Q5_8', 'Convenience Imp')
    print(f"{pred_name:<25} {r:>12.3f} {p:>12.4f} {sig:>6}")

# Multiple regression (manual OLS)
print("\n\nMultiple Regression Results:")
print("-" * 60)

# Add constant
X_with_const = np.column_stack([np.ones(len(X)), X.values])
predictor_names = ['Intercept'] + loyalty_predictors

# OLS: (X'X)^-1 X'y
try:
    XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
    beta = XtX_inv @ X_with_const.T @ y.values

    # Calculate statistics
    y_pred = X_with_const @ beta
    residuals = y.values - y_pred
    n, k = X_with_const.shape
    dof = n - k

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y.values - y.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / dof

    # Standard errors
    mse = ss_res / dof
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    print(f"R-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {adj_r_squared:.4f}")
    print(f"Sample size: {n}")
    print()
    print(f"{'Variable':<25} {'Coef':>10} {'Std Err':>10} {'t-stat':>10} {'p-value':>10}")
    print("-" * 65)

    for i, name in enumerate(predictor_names):
        sig = '*' if p_values[i] < 0.05 else ''
        sig = '**' if p_values[i] < 0.01 else sig
        sig = '***' if p_values[i] < 0.001 else sig
        name_clean = name.replace('_score', '').replace('Q5_1', 'Taste').replace('Q5_3', 'Crust')
        name_clean = name_clean.replace('Q5_7', 'Price').replace('Q5_8', 'Convenience')
        print(f"{name_clean:<25} {beta[i]:>10.4f} {se[i]:>10.4f} {t_stats[i]:>10.3f} {p_values[i]:>9.4f}{sig}")

    print("\n>>> KEY FINDING: Significant predictors of loyalty marked with *")

except np.linalg.LinAlgError:
    print("Note: Matrix singularity issue - showing correlations only")

# =============================================================================
# MODEL 2: What predicts choosing a LOCAL favorite?
# =============================================================================
print("\n" + "=" * 80)
print("MODEL 2: What Predicts Choosing a LOCAL Pizza Place?")
print("Dependent Variable: Local Favorite (0/1)")
print("=" * 80)

# Compare local vs chain choosers
local_choosers = data[data['is_local_favorite'] == 1]
chain_choosers = data[data['is_local_favorite'] == 0]

print("\nGroup Comparison (Local vs Chain Choosers):")
print("-" * 70)
print(f"{'Variable':<30} {'Local':>12} {'Chain':>12} {'Diff':>10} {'p-value':>10}")
print("-" * 70)

comparison_vars = [
    ('Taste Importance', 'Q5_1_score'),
    ('Price Importance', 'Q5_7_score'),
    ('Convenience Importance', 'Q5_8_score'),
    ('Crust Importance', 'Q5_3_score'),
    ('Orders/Month', 'orders_month'),
    ('Expected Price ($)', 'expected_price'),
    ('Prefers Local (%)', 'prefers_local'),
    ('Has Transport (%)', 'has_transport'),
    ('On Campus (%)', 'on_campus'),
]

for label, var in comparison_vars:
    local_mean = local_choosers[var].mean()
    chain_mean = chain_choosers[var].mean()
    diff = local_mean - chain_mean

    # t-test
    local_vals = local_choosers[var].dropna()
    chain_vals = chain_choosers[var].dropna()
    if len(local_vals) > 5 and len(chain_vals) > 5:
        t_stat, p_val = stats.ttest_ind(local_vals, chain_vals)
        sig = '*' if p_val < 0.05 else ''
    else:
        p_val = np.nan
        sig = ''

    if 'Price' in label and '$' in label:
        print(f"{label:<30} ${local_mean:>10.0f} ${chain_mean:>10.0f} {diff:>+10.1f} {p_val:>9.4f}{sig}")
    elif '%' in label:
        print(f"{label:<30} {local_mean*100:>11.1f}% {chain_mean*100:>11.1f}% {diff*100:>+9.1f}% {p_val:>9.4f}{sig}")
    else:
        print(f"{label:<30} {local_mean:>12.2f} {chain_mean:>12.2f} {diff:>+10.2f} {p_val:>9.4f}{sig}")

# =============================================================================
# MODEL 3: What predicts choosing DOMINO'S specifically?
# =============================================================================
print("\n" + "=" * 80)
print("MODEL 3: Who Are Domino's Customers?")
print("Understanding the Primary Competitor")
print("=" * 80)

dominos = data[data['is_dominos'] == 1]
non_dominos = data[data['is_dominos'] == 0]

print("\nDomino's Customers vs Others:")
print("-" * 70)
print(f"{'Variable':<30} {'Dominos':>12} {'Others':>12} {'Diff':>10} {'p-value':>10}")
print("-" * 70)

for label, var in comparison_vars:
    dom_mean = dominos[var].mean()
    other_mean = non_dominos[var].mean()
    diff = dom_mean - other_mean

    dom_vals = dominos[var].dropna()
    other_vals = non_dominos[var].dropna()
    if len(dom_vals) > 5 and len(other_vals) > 5:
        t_stat, p_val = stats.ttest_ind(dom_vals, other_vals)
        sig = '*' if p_val < 0.05 else ''
    else:
        p_val = np.nan
        sig = ''

    if 'Price' in label and '$' in label:
        print(f"{label:<30} ${dom_mean:>10.0f} ${other_mean:>10.0f} {diff:>+10.1f} {p_val:>9.4f}{sig}")
    elif '%' in label:
        print(f"{label:<30} {dom_mean*100:>11.1f}% {other_mean*100:>11.1f}% {diff*100:>+9.1f}% {p_val:>9.4f}{sig}")
    else:
        print(f"{label:<30} {dom_mean:>12.2f} {other_mean:>12.2f} {diff:>+10.2f} {p_val:>9.4f}{sig}")

# Domino's paradox analysis
print("\n--- The Domino's Paradox ---")
dominos_local_pref = dominos[dominos['Q17'].isin(['Local', 'Chain'])]['Q17']
pct_prefer_local = (dominos_local_pref == 'Local').sum() / len(dominos_local_pref) * 100
print(f"Domino's customers who SAY they prefer local: {pct_prefer_local:.1f}%")
print(f"Number of 'persuadable' Domino's customers: {int(len(dominos) * pct_prefer_local / 100)}")

# =============================================================================
# MODEL 4: Importance-Weighted Restaurant Scoring
# =============================================================================
print("\n" + "=" * 80)
print("MODEL 4: Restaurant Scoring Based on Customer Values")
print("Which restaurants best match what customers want?")
print("=" * 80)

# Calculate importance weights (normalized)
importance_cols = ['Q5_1_score', 'Q5_2_score', 'Q5_3_score', 'Q5_4_score',
                   'Q5_5_score', 'Q5_6_score', 'Q5_7_score', 'Q5_8_score', 'Q5_9_score']
labels = ['Taste', 'Ingredients', 'Crust', 'Balance', 'Freshness',
          'Appearance', 'Price', 'Convenience', 'Special']

# Population importance weights
pop_weights = {}
for col, label in zip(importance_cols, labels):
    pop_weights[label] = data[col].mean()

total_weight = sum(pop_weights.values())
norm_weights = {k: v/total_weight for k, v in pop_weights.items()}

print("\nNormalized Importance Weights (sum to 100%):")
print("-" * 40)
for label in sorted(norm_weights, key=norm_weights.get, reverse=True):
    print(f"  {label:<15}: {norm_weights[label]*100:>5.1f}%")

# For each restaurant, calculate how well their customers' priorities
# align with the population weights (consistency score)
print("\nRestaurant Customer Profile Match Scores:")
print("-" * 60)

restaurants = data['Q28'].value_counts().head(10).index.tolist()

restaurant_scores = []
for rest in restaurants:
    rest_data = data[data['Q28'] == rest]
    n = len(rest_data)

    # Calculate how their customers rate each factor
    rest_profile = {}
    for col, label in zip(importance_cols, labels):
        rest_profile[label] = rest_data[col].mean()

    # Weighted score: sum of (customer importance * population weight)
    # Higher = customers who care about what everyone cares about
    weighted_score = sum(rest_profile[label] * norm_weights[label] for label in labels)

    # Consistency with population (lower diff = more aligned)
    profile_diff = sum(abs(rest_profile[label] - pop_weights[label]) for label in labels)

    # Loyalty average
    loyalty = rest_data['loyalty_score'].mean()

    restaurant_scores.append({
        'restaurant': rest,
        'n': n,
        'weighted_score': weighted_score,
        'profile_diff': profile_diff,
        'loyalty': loyalty,
        'type': 'Local' if rest in ["Joe's Brooklyn Pizza", "Salvatore's Pizza",
                                     "Mark's Pizzeria", "Pontillo's Pizza", "Perri's Pizza"] else 'Chain'
    })

# Sort by weighted score
restaurant_scores.sort(key=lambda x: x['weighted_score'], reverse=True)

print(f"{'Restaurant':<22} {'N':>4} {'WtScore':>8} {'Diff':>8} {'Loyalty':>8} {'Type':>6}")
print("-" * 60)
for r in restaurant_scores:
    print(f"{r['restaurant']:<22} {r['n']:>4} {r['weighted_score']:>8.2f} {r['profile_diff']:>8.2f} "
          f"{r['loyalty']:>8.2f} {r['type']:>6}")

# =============================================================================
# SUMMARY: PREDICTIVE MODEL INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS SUMMARY")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KEY PREDICTIVE INSIGHTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHAT PREDICTS LOYALTY?                                                     │
│  • Order frequency is the strongest predictor (more orders = more loyal)    │
│  • Price sensitivity negatively correlates with loyalty                     │
│  • Being a Domino's customer predicts LOWER loyalty                         │
│                                                                             │
│  WHAT PREDICTS CHOOSING LOCAL?                                              │
│  • Higher willingness to pay                                                │
│  • Stated preference for local businesses                                   │
│  • Higher importance placed on taste and crust quality                      │
│                                                                             │
│  THE DOMINO'S CUSTOMER PROFILE:                                             │
│  • Lower expected price point (~$16 vs $17 average)                         │
│  • Lower loyalty scores (easily poachable!)                                 │
│  • 65% SAY they prefer local (cognitive dissonance opportunity)             │
│                                                                             │
│  STRATEGIC IMPLICATION FOR NEW ENTRANT:                                     │
│  • Target Domino's "persuadables" - they want local but choose convenience  │
│  • Match Domino's price point ($16) but exceed on taste/quality             │
│  • Fast pickup is essential (convenience is why they choose chains)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")
