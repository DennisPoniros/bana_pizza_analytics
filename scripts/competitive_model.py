"""
BANA255 Pizza Survey - Competitive Analytics Model
====================================================
Objective: Build a data-driven model to understand WHY students perceive their
favorite pizza as "the best" and identify the key competitor for a new local entrant.

Analytical Framework:
1. Weighted Importance Model - What factors define "best"?
2. Customer Segmentation - Who are the target customer profiles?
3. Restaurant Profiling - Which restaurants capture which segments?
4. Predictive Model - What predicts restaurant preference?
5. Competitive Ranking - Who is the real competition?
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================
print("=" * 80)
print("COMPETITIVE ANALYTICS MODEL: Best Pizza Analysis")
print("=" * 80)

xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
questions = df.iloc[0].to_dict()
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

print(f"Sample Size: {len(data)} respondents")
print()

# Importance mapping
importance_map = {
    'Not at all important': 1, 'Slightly important': 2,
    'Moderately important': 3, 'Very important': 4, 'Extremely important': 5
}

likelihood_map = {
    'Extremely unlikely': 1, 'Somewhat unlikely': 2,
    'Neither likely nor unlikely': 3, 'Somewhat likely': 4, 'Extremely likely': 5
}

# Convert key columns to numeric
q5_cols = ['Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5', 'Q5_6', 'Q5_7', 'Q5_8', 'Q5_9']
q5_labels = {
    'Q5_1': 'Taste', 'Q5_2': 'Ingredients', 'Q5_3': 'Crust',
    'Q5_4': 'Balance', 'Q5_5': 'Freshness', 'Q5_6': 'Appearance',
    'Q5_7': 'Price', 'Q5_8': 'Convenience', 'Q5_9': 'Special Features'
}

for col in q5_cols:
    data[f'{col}_score'] = data[col].map(importance_map)

data['Q4_num'] = pd.to_numeric(data['Q4'], errors='coerce')
data['Q29_score'] = data['Q29'].map({
    'Not loyal': 1, 'Slightly loyal': 2, 'Moderately loyal': 3,
    'Very loyal': 4, 'Extremely loyal': 5
})
data['Q21_1_num'] = pd.to_numeric(data['Q21_1'], errors='coerce')
data['Q21_2_num'] = pd.to_numeric(data['Q21_2'], errors='coerce')

# =============================================================================
# PART 1: WEIGHTED IMPORTANCE MODEL
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: WEIGHTED IMPORTANCE MODEL")
print("What factors define 'the best' pizza?")
print("=" * 80)

# Calculate population-level importance weights
importance_weights = {}
for col in q5_cols:
    label = q5_labels[col]
    scores = data[f'{col}_score'].dropna()
    importance_weights[label] = {
        'mean': scores.mean(),
        'weight': scores.mean() / 5,  # Normalize to 0-1
        'pct_high': ((scores >= 4).sum() / len(scores)) * 100
    }

# Sort by importance
sorted_weights = sorted(importance_weights.items(), key=lambda x: x[1]['mean'], reverse=True)

print("\nImportance Weights (Population-Level):")
print("-" * 60)
print(f"{'Factor':<20} {'Mean':>8} {'Weight':>8} {'% High':>10}")
print("-" * 60)
total_weight = sum(w['mean'] for _, w in sorted_weights)
for factor, w in sorted_weights:
    norm_weight = w['mean'] / total_weight * 100
    print(f"{factor:<20} {w['mean']:>8.2f} {norm_weight:>7.1f}% {w['pct_high']:>9.1f}%")

print("\n>>> KEY INSIGHT: Normalized weight distribution for 'best pizza' definition:")
print("    This becomes our SCORING MODEL for evaluating any pizza place.")

# Create composite importance score for each respondent
score_cols = [f'{col}_score' for col in q5_cols]
data['importance_profile'] = data[score_cols].mean(axis=1)

# =============================================================================
# PART 2: CUSTOMER SEGMENTATION
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: CUSTOMER SEGMENTATION")
print("Who are the distinct customer profiles?")
print("=" * 80)

# Prepare clustering data
cluster_features = data[score_cols].dropna()
cluster_idx = cluster_features.index

# Hierarchical clustering
Z = linkage(cluster_features, method='ward')
n_clusters = 4
data.loc[cluster_idx, 'segment'] = fcluster(Z, n_clusters, criterion='maxclust')

# Analyze segments
print("\nCustomer Segments Identified:")
print("-" * 70)

segment_profiles = {}
for seg in range(1, n_clusters + 1):
    seg_data = data[data['segment'] == seg]
    n = len(seg_data)

    # Calculate segment characteristics
    profile = {
        'n': n,
        'pct': n / len(data[data['segment'].notna()]) * 100
    }

    for col in q5_cols:
        label = q5_labels[col]
        profile[label] = seg_data[f'{col}_score'].mean()

    # Top restaurant
    top_restaurant = seg_data['Q28'].value_counts().head(1)
    profile['top_restaurant'] = top_restaurant.index[0] if len(top_restaurant) > 0 else 'N/A'
    profile['top_rest_pct'] = top_restaurant.values[0] / n * 100 if len(top_restaurant) > 0 else 0

    # Avg order frequency
    profile['orders_month'] = seg_data['Q4_num'].mean()

    # Local preference
    local_pref = seg_data[seg_data['Q17'].isin(['Local', 'Chain'])]['Q17']
    profile['local_pref'] = (local_pref == 'Local').sum() / len(local_pref) * 100 if len(local_pref) > 0 else 50

    segment_profiles[seg] = profile

# Name segments based on characteristics
segment_names = {}
for seg, prof in segment_profiles.items():
    if prof['Taste'] > 4.5 and prof['Price'] > 4:
        segment_names[seg] = "Quality Seekers"
    elif prof['Price'] > 4 and prof['Convenience'] > 3.5:
        segment_names[seg] = "Value Hunters"
    elif prof['Convenience'] > 3.5 and prof['Special Features'] > 2.5:
        segment_names[seg] = "Convenience First"
    else:
        segment_names[seg] = "Balanced Buyers"

# Reassign to ensure distinct names
name_counts = Counter(segment_names.values())
for seg in segment_names:
    if name_counts[segment_names[seg]] > 1:
        prof = segment_profiles[seg]
        if prof['local_pref'] > 80:
            segment_names[seg] = "Local Loyalists"
        elif prof['orders_month'] > 3:
            segment_names[seg] = "Frequent Orderers"

for seg, prof in segment_profiles.items():
    name = segment_names[seg]
    print(f"\n--- Segment {seg}: {name} ({prof['n']} respondents, {prof['pct']:.1f}%) ---")
    print(f"    Top Restaurant: {prof['top_restaurant']} ({prof['top_rest_pct']:.0f}%)")
    print(f"    Orders/Month: {prof['orders_month']:.1f}")
    print(f"    Local Preference: {prof['local_pref']:.0f}%")
    print(f"    Key Priorities:")
    priorities = [(k, v) for k, v in prof.items() if k in q5_labels.values()]
    priorities.sort(key=lambda x: x[1], reverse=True)
    for p, v in priorities[:3]:
        print(f"      - {p}: {v:.2f}")

# =============================================================================
# PART 3: RESTAURANT PROFILING
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: RESTAURANT COMPETITIVE PROFILING")
print("Which restaurants capture which customers?")
print("=" * 80)

# Analyze top restaurants
top_restaurants = data['Q28'].value_counts().head(10).index.tolist()

restaurant_profiles = {}
for restaurant in top_restaurants:
    rest_data = data[data['Q28'] == restaurant]
    n = len(rest_data)

    profile = {
        'n': n,
        'share': n / len(data) * 100,
        'type': 'Chain' if restaurant in ['Domino\'s Pizza', 'Papa John\'s', 'Little Caesars',
                                           'Pizza Hut', 'Costco Pizza', 'Blaze Pizza'] else 'Local'
    }

    # Customer characteristics
    profile['avg_orders'] = rest_data['Q4_num'].mean()
    profile['loyalty'] = rest_data['Q29_score'].mean()
    profile['expected_price'] = rest_data['Q21_1_num'].mean()
    profile['max_price'] = rest_data['Q21_2_num'].mean()

    # What their customers value (importance scores)
    for col in q5_cols:
        label = q5_labels[col]
        profile[f'imp_{label}'] = rest_data[f'{col}_score'].mean()

    # Local preference of their customers
    lp = rest_data[rest_data['Q17'].isin(['Local', 'Chain'])]['Q17']
    profile['customers_prefer_local'] = (lp == 'Local').sum() / len(lp) * 100 if len(lp) > 0 else 50

    # Segment distribution
    seg_dist = rest_data['segment'].value_counts(normalize=True) * 100
    profile['segments'] = seg_dist.to_dict()

    restaurant_profiles[restaurant] = profile

print("\nRestaurant Competitive Matrix:")
print("-" * 90)
print(f"{'Restaurant':<22} {'N':>4} {'Share':>6} {'Type':<6} {'Loyalty':>7} {'Exp.$':>6} {'Taste':>6} {'Price':>6}")
print("-" * 90)

for rest in top_restaurants:
    p = restaurant_profiles[rest]
    print(f"{rest:<22} {p['n']:>4} {p['share']:>5.1f}% {p['type']:<6} {p['loyalty']:>7.2f} "
          f"${p['expected_price']:>5.0f} {p['imp_Taste']:>6.2f} {p['imp_Price']:>6.2f}")

# =============================================================================
# PART 4: PREDICTIVE MODEL - What predicts restaurant preference?
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: PREDICTIVE MODEL")
print("What factors predict choosing each restaurant?")
print("=" * 80)

# Focus on top 5 restaurants for meaningful analysis
top5 = data['Q28'].value_counts().head(5).index.tolist()

print("\nKey Differentiators by Restaurant (vs. population mean):")
print("-" * 70)

population_means = {q5_labels[col]: data[f'{col}_score'].mean() for col in q5_cols}
population_means['orders_month'] = data['Q4_num'].mean()
population_means['expected_price'] = data['Q21_1_num'].mean()

for restaurant in top5:
    rest_data = data[data['Q28'] == restaurant]
    print(f"\n{restaurant}:")

    differentiators = []
    for col in q5_cols:
        label = q5_labels[col]
        rest_mean = rest_data[f'{col}_score'].mean()
        pop_mean = population_means[label]
        diff = rest_mean - pop_mean
        if abs(diff) > 0.15:  # Meaningful difference threshold
            differentiators.append((label, diff, rest_mean))

    # Also check behavioral differences
    rest_orders = rest_data['Q4_num'].mean()
    if abs(rest_orders - population_means['orders_month']) > 0.3:
        differentiators.append(('Order Frequency', rest_orders - population_means['orders_month'], rest_orders))

    rest_price = rest_data['Q21_1_num'].mean()
    if abs(rest_price - population_means['expected_price']) > 1:
        differentiators.append(('Price Expectation', rest_price - population_means['expected_price'], rest_price))

    differentiators.sort(key=lambda x: abs(x[1]), reverse=True)

    if differentiators:
        for label, diff, val in differentiators[:4]:
            direction = "+" if diff > 0 else ""
            print(f"  • {label}: {val:.2f} ({direction}{diff:.2f} vs avg)")
    else:
        print("  • No significant differentiators (matches average profile)")

# =============================================================================
# PART 5: COMPOSITE COMPETITIVE RANKING
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: COMPOSITE COMPETITIVE RANKING")
print("Who is the REAL competition for a new local entrant?")
print("=" * 80)

# Build composite score based on:
# 1. Market share (how many customers)
# 2. Customer loyalty (retention potential)
# 3. Match to "ideal" importance profile
# 4. Local-preferring customer capture

# Calculate "ideal" pizza profile (population importance means)
ideal_profile = {q5_labels[col]: data[f'{col}_score'].mean() for col in q5_cols}

print("\nScoring Methodology:")
print("  - Market Share Score (30%): Current customer base")
print("  - Loyalty Score (25%): Customer retention strength")
print("  - Profile Match Score (25%): Alignment with 'ideal pizza' factors")
print("  - Local Capture Score (20%): Success with local-preferring customers")

competitive_scores = {}
for restaurant in top_restaurants:
    p = restaurant_profiles[restaurant]

    # Market share score (normalize to 0-100)
    max_share = max(rp['share'] for rp in restaurant_profiles.values())
    share_score = (p['share'] / max_share) * 100

    # Loyalty score (normalize 1-5 to 0-100)
    loyalty_score = ((p['loyalty'] - 1) / 4) * 100 if not np.isnan(p['loyalty']) else 50

    # Profile match score (how well do their customers' values match ideal)
    profile_diffs = []
    for col in q5_cols:
        label = q5_labels[col]
        customer_imp = p[f'imp_{label}']
        ideal_imp = ideal_profile[label]
        # Closer to ideal = better score
        profile_diffs.append(abs(customer_imp - ideal_imp))
    avg_diff = np.mean(profile_diffs)
    profile_score = max(0, 100 - (avg_diff * 25))  # Convert diff to score

    # Local capture score
    local_capture = p['customers_prefer_local']

    # Composite score
    composite = (share_score * 0.30 +
                 loyalty_score * 0.25 +
                 profile_score * 0.25 +
                 local_capture * 0.20)

    competitive_scores[restaurant] = {
        'composite': composite,
        'share_score': share_score,
        'loyalty_score': loyalty_score,
        'profile_score': profile_score,
        'local_capture': local_capture,
        'type': p['type']
    }

# Rank by composite score
ranked = sorted(competitive_scores.items(), key=lambda x: x[1]['composite'], reverse=True)

print("\nCOMPETITIVE RANKING (Threat Level for New Local Entrant):")
print("-" * 85)
print(f"{'Rank':<5} {'Restaurant':<22} {'Type':<6} {'Share':>7} {'Loyal':>7} {'Match':>7} {'Local':>7} {'TOTAL':>8}")
print("-" * 85)

for rank, (restaurant, scores) in enumerate(ranked, 1):
    print(f"{rank:<5} {restaurant:<22} {scores['type']:<6} {scores['share_score']:>6.1f} "
          f"{scores['loyalty_score']:>6.1f} {scores['profile_score']:>6.1f} "
          f"{scores['local_capture']:>6.1f} {scores['composite']:>7.1f}")

# Identify primary competitor
winner = ranked[0]
print(f"\n{'='*85}")
print(f"PRIMARY COMPETITOR: {winner[0]}")
print(f"{'='*85}")
print(f"Composite Threat Score: {winner[1]['composite']:.1f}/100")

# Identify top LOCAL competitor
local_ranked = [(r, s) for r, s in ranked if s['type'] == 'Local']
if local_ranked:
    local_winner = local_ranked[0]
    print(f"\nTOP LOCAL COMPETITOR: {local_winner[0]}")
    print(f"Composite Threat Score: {local_winner[1]['composite']:.1f}/100")

# =============================================================================
# PART 6: STRATEGIC RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: STRATEGIC RECOMMENDATIONS FOR NEW ENTRANT")
print("=" * 80)

# Calculate opportunity gaps
print("\n1. POSITIONING STRATEGY:")
print("-" * 50)

# The "Local Paradox" opportunity
dominos_data = data[data['Q28'] == "Domino's Pizza"]
dominos_local_pref = dominos_data[dominos_data['Q17'].isin(['Local', 'Chain'])]['Q17']
dominos_local_pct = (dominos_local_pref == 'Local').sum() / len(dominos_local_pref) * 100

print(f"   • {dominos_local_pct:.0f}% of Domino's customers SAY they prefer local")
print(f"   • This represents ~{int(len(dominos_data) * dominos_local_pct / 100)} 'persuadable' customers")
print(f"   • OPPORTUNITY: Capture customers who want local but default to chains")

print("\n2. PRODUCT PRIORITIES (Based on Importance Weights):")
print("-" * 50)
for i, (factor, w) in enumerate(sorted_weights[:5], 1):
    print(f"   {i}. {factor}: {w['mean']:.2f}/5 ({w['pct_high']:.0f}% rate highly important)")

print("\n3. PRICING STRATEGY:")
print("-" * 50)
avg_expected = data['Q21_1_num'].mean()
avg_max = data['Q21_2_num'].mean()
print(f"   • Average expected price (16\" pizza): ${avg_expected:.0f}")
print(f"   • Maximum for 'the best': ${avg_max:.0f}")
print(f"   • SWEET SPOT: ${avg_expected:.0f}-${avg_max:.0f} range")
print(f"   • Premium positioning possible at ${avg_max:.0f} IF quality delivers")

print("\n4. SERVICE MODEL:")
print("-" * 50)
pickup_pct = (data['Q11'] == 'Pick up').sum() / len(data) * 100
delivery_pct = (data['Q11'] == 'Delivery').sum() / len(data) * 100
print(f"   • {pickup_pct:.0f}% prefer pickup → Prioritize efficient pickup operations")
print(f"   • {delivery_pct:.0f}% prefer delivery → Offer but don't over-invest")
print(f"   • Expected pickup time: {data['Q14_2'].apply(pd.to_numeric, errors='coerce').mean():.0f} min")

print("\n5. TARGET SEGMENTS:")
print("-" * 50)
# Find segments with highest local preference
for seg, prof in sorted(segment_profiles.items(), key=lambda x: x[1]['local_pref'], reverse=True)[:2]:
    name = segment_names[seg]
    print(f"   • {name}: {prof['local_pref']:.0f}% prefer local, {prof['n']} customers")
    print(f"     - Top priority: {sorted([(k,v) for k,v in prof.items() if k in q5_labels.values()], key=lambda x:x[1], reverse=True)[0][0]}")

print("\n6. COMPETITIVE DIFFERENTIATION:")
print("-" * 50)
# What gaps exist?
print(f"   vs. Domino's (main threat):")
print(f"      - Domino's wins on: Convenience, Price, Speed")
print(f"      - Gap to exploit: Taste quality, Fresh ingredients, 'Local feel'")
if local_ranked:
    local_top = local_ranked[0][0]
    print(f"   vs. {local_top} (top local competitor):")
    lp = restaurant_profiles[local_top]
    print(f"      - They capture: {lp['share']:.1f}% market share")
    print(f"      - Their customer loyalty: {lp['loyalty']:.2f}/5")

# =============================================================================
# SUMMARY OUTPUT
# =============================================================================
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY: COMPETITIVE MODEL FINDINGS")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODEL CONCLUSIONS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. "BEST PIZZA" DEFINITION (Weighted Importance Model):                    │
│     Taste (15.2%) > Balance (13.6%) > Crust (13.2%) > Freshness (13.1%)    │
│     > Price (13.0%) > Ingredients (12.0%) > Convenience (10.8%)             │
│     > Appearance (9.4%) > Special Features (7.8%)                           │
│                                                                             │
│  2. PRIMARY COMPETITOR: {winner_name:<40}         │
│     - Threat Score: {threat_score:.1f}/100                                              │
│     - Why: Dominates market share despite customers preferring local        │
│                                                                             │
│  3. TOP LOCAL COMPETITOR: {local_winner_name:<40}     │
│     - Threat Score: {local_threat:.1f}/100                                              │
│     - Why: Proven local success model to study/differentiate from           │
│                                                                             │
│  4. KEY OPPORTUNITY: {opp_customers} customers at Domino's prefer local but haven't    │
│     switched → Target with quality + competitive pricing                    │
│                                                                             │
│  5. WINNING FORMULA FOR NEW ENTRANT:                                        │
│     ✓ Exceptional taste (non-negotiable)                                    │
│     ✓ Price point: ${price_low:.0f}-${price_high:.0f} for 16" pizza                               │
│     ✓ Fast pickup-focused service model                                     │
│     ✓ Position as "local quality at chain convenience"                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""".format(
    winner_name=winner[0],
    threat_score=winner[1]['composite'],
    local_winner_name=local_winner[0] if local_ranked else "N/A",
    local_threat=local_winner[1]['composite'] if local_ranked else 0,
    opp_customers=int(len(dominos_data) * dominos_local_pct / 100),
    price_low=avg_expected,
    price_high=avg_max
))
