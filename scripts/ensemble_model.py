"""
BANA255 Pizza Survey - Ensemble Learning Model (v2)
====================================================

PURPOSE:
--------
This model answers the question: "Given a student's pizza preferences, ordering
behaviors, and demographics - can we predict whether they will choose a LOCAL
or CHAIN restaurant?"

This is valuable for the new local entrant because it identifies:
1. Which customer characteristics predict local restaurant choice
2. What behavioral factors differentiate local vs chain customers
3. Actionable features the business can target in marketing

IMPORTANT: We deliberately EXCLUDE "states_prefer_local" as a feature because:
- It creates circular logic (stated preference predicting behavior is tautological)
- The interesting question is: what ELSE predicts choice beyond stated preference?
- This reveals the "persuadable" segment: people whose behavior doesn't match preference

TARGET VARIABLE:
- chose_local: 1 if favorite restaurant is local, 0 if chain

FEATURES (28 total, grouped by business relevance):
- Pizza quality importance (what they value in pizza)
- Ordering behavior (how they order)
- Time preferences (patience/expectations)
- Price sensitivity (budget and flexibility)
- Demographics (who they are)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 80)
print("ENSEMBLE LEARNING MODEL: Predicting Local vs Chain Choice")
print("=" * 80)
print("""
PURPOSE: Identify what characteristics predict whether a student chooses
         a LOCAL restaurant vs a CHAIN - without using their stated preference.

WHY THIS MATTERS: Reveals actionable factors for customer acquisition.
""")

xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

print(f"Sample Size: {len(data)} students")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
importance_map = {
    'Not at all important': 1, 'Slightly important': 2,
    'Moderately important': 3, 'Very important': 4, 'Extremely important': 5
}

likelihood_map = {
    'Extremely unlikely': 1, 'Somewhat unlikely': 2,
    'Neither likely nor unlikely': 3, 'Somewhat likely': 4, 'Extremely likely': 5
}

# === FEATURE GROUP 1: Pizza Quality Importance ===
# Business relevance: What do customers VALUE in pizza?
quality_features = {
    'imp_taste': 'Q5_1',           # Taste & flavor
    'imp_ingredients': 'Q5_2',     # Ingredient quality
    'imp_crust': 'Q5_3',           # Crust excellence
    'imp_balance': 'Q5_4',         # Sauce/cheese/topping balance
    'imp_freshness': 'Q5_5',       # Freshness & temperature
    'imp_appearance': 'Q5_6',      # Visual appeal
    'imp_price': 'Q5_7',           # Price & value
    'imp_convenience': 'Q5_8',     # Quick service
    'imp_special': 'Q5_9'          # Special features
}

for feat_name, col in quality_features.items():
    data[feat_name] = data[col].map(importance_map)

# === FEATURE GROUP 2: Ordering Behavior ===
# Business relevance: HOW do they order? (Actionable for service design)
data['orders_per_month'] = pd.to_numeric(data['Q4'], errors='coerce')
data['prefers_pickup'] = (data['Q11'] == 'Pick up').astype(int)
data['prefers_delivery'] = (data['Q11'] == 'Delivery').astype(int)
data['orders_online'] = (data['Q12'] == 'Online (any method)').astype(int)

# === FEATURE GROUP 3: Time Preferences ===
# Business relevance: How patient are they? (Affects service model)
data['expected_delivery_time'] = pd.to_numeric(data['Q14_1'], errors='coerce')
data['expected_pickup_time'] = pd.to_numeric(data['Q14_2'], errors='coerce')
data['willing_wait_delivery'] = pd.to_numeric(data['Q15_1'], errors='coerce')
data['willing_drive_pickup'] = pd.to_numeric(data['Q15_2'], errors='coerce')

# === FEATURE GROUP 4: Price Sensitivity ===
# Business relevance: What's their budget? (Affects pricing strategy)
data['expected_price'] = pd.to_numeric(data['Q21_1'], errors='coerce')
data['max_price'] = pd.to_numeric(data['Q21_2'], errors='coerce')
data['price_flexibility'] = data['max_price'] - data['expected_price']
data['price_over_location'] = (data['Q16'] == 'Price').astype(int)

# === FEATURE GROUP 5: Demographics ===
# Business relevance: WHO are the target customers?
data['age'] = pd.to_numeric(data['Q30'], errors='coerce')
data['has_transport'] = (data['Q36'] == 'Yes').astype(int)
data['on_campus'] = (data['Q33'] == 'On campus').astype(int)
data['off_campus'] = (data['Q33'] == 'Off campus').astype(int)

year_map = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4,
            'Super Senior': 5, 'Graduate student': 6}
data['year_numeric'] = data['Q31'].map(year_map)

# === FEATURE GROUP 6: Additional Preferences ===
data['imp_topping_variety'] = data['Q8'].map(importance_map)
data['imp_foldability'] = data['Q7'].map({
    'Extremely important': 5, 'Very important': 4, 'Moderately important': 3,
    'Slightly important': 2, 'Not at all important': 1
})
data['deal_sensitivity'] = data['Q19'].map(likelihood_map)

# === TARGET VARIABLE ===
local_restaurants = ["Joe's Brooklyn Pizza", "Salvatore's Pizza", "Mark's Pizzeria",
                     "Pontillo's Pizza", "Perri's Pizza", "Fire Crust Pizza",
                     "Peels on Wheels", "Brandani's Pizza", "Tony Pepperoni",
                     "Pizza Wizard", "East of Chicago"]

data['chose_local'] = data['Q28'].isin(local_restaurants).astype(int)

# =============================================================================
# FEATURE SELECTION (NO CIRCULAR FEATURES)
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# IMPORTANT: We EXCLUDE states_prefer_local to avoid circular logic
feature_cols = [
    # Quality importance (what they value)
    'imp_taste', 'imp_ingredients', 'imp_crust', 'imp_balance',
    'imp_freshness', 'imp_appearance', 'imp_price', 'imp_convenience', 'imp_special',
    # Ordering behavior (how they order)
    'orders_per_month', 'prefers_pickup', 'prefers_delivery', 'orders_online',
    # Time preferences (patience)
    'expected_delivery_time', 'expected_pickup_time',
    'willing_wait_delivery', 'willing_drive_pickup',
    # Price sensitivity (budget)
    'expected_price', 'max_price', 'price_flexibility', 'price_over_location',
    # Demographics (who they are)
    'age', 'has_transport', 'on_campus', 'year_numeric',
    # Other preferences
    'imp_topping_variety', 'imp_foldability', 'deal_sensitivity'
]

print(f"Features used: {len(feature_cols)}")
print("\nNOTE: 'states_prefer_local' is EXCLUDED to avoid circular logic.")
print("      We want to predict behavior from characteristics, not from stated preference.")

# Prepare data
model_data = data[feature_cols + ['chose_local']].dropna()
X = model_data[feature_cols]
y = model_data['chose_local']

print(f"\nSamples after dropping missing: {len(X)}")
print(f"Target: Chose Local={y.sum()} ({y.mean()*100:.1f}%), Chose Chain={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# MODEL TRAINING
# =============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

# Models
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_split=10,
    min_samples_leaf=5, random_state=42, class_weight='balanced'
)

gb_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    min_samples_split=10, random_state=42
)

lr_model = LogisticRegression(
    C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
    class_weight='balanced', random_state=42
)

dt_model = DecisionTreeClassifier(
    max_depth=4, min_samples_split=15, min_samples_leaf=10,
    random_state=42, class_weight='balanced'
)

# Cross-validation
print("\n5-Fold Cross-Validation:")
print("-" * 50)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest': (rf_model, X_train),
    'Gradient Boosting': (gb_model, X_train),
    'Logistic Regression': (lr_model, X_train_scaled),
    'Decision Tree': (dt_model, X_train)
}

cv_results = {}
for name, (model, X_cv) in models.items():
    scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = scores
    print(f"  {name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lr_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train, y_train)

# =============================================================================
# MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION (Test Set)")
print("=" * 80)

results = {
    'Random Forest': (rf_model.predict(X_test), rf_model.predict_proba(X_test)[:, 1]),
    'Gradient Boosting': (gb_model.predict(X_test), gb_model.predict_proba(X_test)[:, 1]),
    'Logistic Regression': (lr_model.predict(X_test_scaled), lr_model.predict_proba(X_test_scaled)[:, 1]),
    'Decision Tree': (dt_model.predict(X_test), dt_model.predict_proba(X_test)[:, 1])
}

print(f"\n{'Model':<22} {'Accuracy':>10} {'AUC-ROC':>10}")
print("-" * 45)
for name, (pred, proba) in results.items():
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    print(f"{name:<22} {acc:>10.1%} {auc:>10.3f}")

# Best model
best_name = max(results.keys(), key=lambda k: accuracy_score(y_test, results[k][0]))
best_pred, best_proba = results[best_name]
best_acc = accuracy_score(y_test, best_pred)
best_auc = roc_auc_score(y_test, best_proba)

print(f"\nBest Model: {best_name} (Accuracy: {best_acc:.1%}, AUC: {best_auc:.3f})")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("What characteristics predict choosing LOCAL?")
print("=" * 80)

# Random Forest importance
rf_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 10 Features (Random Forest) ---")
print(f"{'Rank':<5} {'Feature':<30} {'Importance':>12}")
print("-" * 50)
for i, (_, row) in enumerate(rf_imp.head(10).iterrows(), 1):
    print(f"{i:<5} {row['feature']:<30} {row['importance']:>12.4f}")

# Logistic Regression coefficients (direction matters)
lr_coef = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\n--- Top 10 Features (Logistic Regression) ---")
print(f"{'Rank':<5} {'Feature':<30} {'Coefficient':>12} {'Direction':>12}")
print("-" * 65)
for i, (_, row) in enumerate(lr_coef.head(10).iterrows(), 1):
    direction = "→ LOCAL" if row['coefficient'] > 0 else "→ CHAIN"
    print(f"{i:<5} {row['feature']:<30} {row['coefficient']:>+12.3f} {direction:>12}")

# Permutation importance (most robust)
print("\n--- Permutation Importance (Most Robust) ---")
perm_imp = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)
perm_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': perm_imp.importances_mean,
    'std': perm_imp.importances_std
}).sort_values('importance', ascending=False)

print(f"{'Rank':<5} {'Feature':<30} {'Importance':>12}")
print("-" * 50)
for i, (_, row) in enumerate(perm_df.head(10).iterrows(), 1):
    print(f"{i:<5} {row['feature']:<30} {row['importance']:>12.4f}")

# =============================================================================
# CONSENSUS RANKING
# =============================================================================
print("\n" + "=" * 80)
print("CONSENSUS FEATURE RANKING")
print("=" * 80)

# Normalize and average
rf_norm = rf_imp.set_index('feature')['importance'] / rf_imp['importance'].max()
gb_imp = pd.DataFrame({'feature': feature_cols, 'importance': gb_model.feature_importances_})
gb_norm = gb_imp.set_index('feature')['importance'] / gb_imp['importance'].max()
lr_norm = lr_coef.set_index('feature')['coefficient'].abs() / lr_coef['coefficient'].abs().max()
perm_norm = perm_df.set_index('feature')['importance']
perm_norm = (perm_norm - perm_norm.min()) / (perm_norm.max() - perm_norm.min() + 0.001)

consensus = pd.DataFrame({'RF': rf_norm, 'GB': gb_norm, 'LR': lr_norm, 'Perm': perm_norm})
consensus['mean'] = consensus.mean(axis=1)
consensus = consensus.sort_values('mean', ascending=False)

print(f"\n{'Rank':<5} {'Feature':<30} {'Consensus Score':>15}")
print("-" * 55)
for i, (feat, row) in enumerate(consensus.head(10).iterrows(), 1):
    print(f"{i:<5} {feat:<30} {row['mean']:>15.3f}")

# Save for visualization
consensus.to_csv('outputs/feature_importance_consensus.csv')

# =============================================================================
# BUSINESS INTERPRETATION
# =============================================================================
print("\n" + "=" * 80)
print("BUSINESS INTERPRETATION")
print("=" * 80)

top_features = consensus.head(5).index.tolist()
print(f"""
KEY PREDICTORS OF LOCAL CHOICE (without using stated preference):

1. {top_features[0].upper().replace('_', ' ')}
2. {top_features[1].upper().replace('_', ' ')}
3. {top_features[2].upper().replace('_', ' ')}
4. {top_features[3].upper().replace('_', ' ')}
5. {top_features[4].upper().replace('_', ' ')}

MODEL ACCURACY: {best_acc:.1%} (vs 50% random baseline)
                {best_acc - 0.5:.1%} improvement over random guessing

INTERPRETATION:
Without knowing whether someone "says" they prefer local, we can still
predict their choice with {best_acc:.0%} accuracy using behavioral factors.

The model reveals that LOCAL choosers:
""")

# Compare feature means
local_means = X_test[y_test == 1].mean()
chain_means = X_test[y_test == 0].mean()

for feat in top_features[:5]:
    diff = local_means[feat] - chain_means[feat]
    if diff > 0:
        print(f"  • Have HIGHER {feat.replace('_', ' ')}")
    else:
        print(f"  • Have LOWER {feat.replace('_', ' ')}")

print(f"""
ACTIONABLE INSIGHTS FOR NEW LOCAL ENTRANT:
These are the characteristics to target in marketing and service design,
because they predict LOCAL preference independent of stated preference.
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)
print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE MODEL RESULTS (v2 - No Circular Features)        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PURPOSE: Predict local vs chain choice WITHOUT using stated preference     │
│                                                                             │
│  BEST MODEL: {best_name:<50}       │
│  ACCURACY: {best_acc:.1%} (baseline: 50%, improvement: {best_acc-0.5:.1%})                      │
│  AUC-ROC: {best_auc:.3f}                                                            │
│                                                                             │
│  TOP PREDICTIVE FEATURES (behavioral, not stated):                          │
│    1. {consensus.index[0]:<55}        │
│    2. {consensus.index[1]:<55}        │
│    3. {consensus.index[2]:<55}        │
│    4. {consensus.index[3]:<55}        │
│    5. {consensus.index[4]:<55}        │
│                                                                             │
│  BUSINESS VALUE:                                                            │
│  These features identify "persuadable" customers - people whose             │
│  behavioral characteristics match LOCAL choosers but may currently          │
│  choose chains due to convenience. Target them with marketing.              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")
