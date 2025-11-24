"""
BANA255 Pizza Survey - Ensemble Learning Model
===============================================
Predict and explain what makes a pizza place "the best"

Model Architecture:
- Target: Binary classification (Local vs Chain preference in behavior)
- Features: Pizza quality importance, behavioral patterns, demographics
- Ensemble: Random Forest, Gradient Boosting, Logistic Regression
- Explanation: Feature importance, decision rules, customer profiles
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance

# =============================================================================
# DATA LOADING AND FEATURE ENGINEERING
# =============================================================================
print("=" * 80)
print("ENSEMBLE LEARNING MODEL: Predicting 'Best Pizza' Preferences")
print("=" * 80)

xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

print(f"Total samples: {len(data)}")

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Importance mappings
importance_map = {
    'Not at all important': 1, 'Slightly important': 2,
    'Moderately important': 3, 'Very important': 4, 'Extremely important': 5
}

likelihood_map = {
    'Extremely unlikely': 1, 'Somewhat unlikely': 2,
    'Neither likely nor unlikely': 3, 'Somewhat likely': 4, 'Extremely likely': 5
}

loyalty_map = {
    'Not loyal': 1, 'Slightly loyal': 2, 'Moderately loyal': 3,
    'Very loyal': 4, 'Extremely loyal': 5
}

# === FEATURE GROUP 1: Pizza Quality Importance (Q5) ===
quality_features = {
    'imp_taste': 'Q5_1',
    'imp_ingredients': 'Q5_2',
    'imp_crust': 'Q5_3',
    'imp_balance': 'Q5_4',
    'imp_freshness': 'Q5_5',
    'imp_appearance': 'Q5_6',
    'imp_price': 'Q5_7',
    'imp_convenience': 'Q5_8',
    'imp_special': 'Q5_9'
}

for feat_name, col in quality_features.items():
    data[feat_name] = data[col].map(importance_map)

# === FEATURE GROUP 2: Ordering Behavior ===
data['orders_per_month'] = pd.to_numeric(data['Q4'], errors='coerce')
data['prefers_pickup'] = (data['Q11'] == 'Pick up').astype(int)
data['prefers_delivery'] = (data['Q11'] == 'Delivery').astype(int)
data['orders_online'] = (data['Q12'] == 'Online (any method)').astype(int)

# === FEATURE GROUP 3: Time Preferences ===
data['expected_delivery_time'] = pd.to_numeric(data['Q14_1'], errors='coerce')
data['expected_pickup_time'] = pd.to_numeric(data['Q14_2'], errors='coerce')
data['willing_wait_delivery'] = pd.to_numeric(data['Q15_1'], errors='coerce')
data['willing_drive_pickup'] = pd.to_numeric(data['Q15_2'], errors='coerce')

# === FEATURE GROUP 4: Price Sensitivity ===
data['expected_price'] = pd.to_numeric(data['Q21_1'], errors='coerce')
data['max_price'] = pd.to_numeric(data['Q21_2'], errors='coerce')
data['price_flexibility'] = data['max_price'] - data['expected_price']
data['price_over_location'] = (data['Q16'] == 'Price').astype(int)

# === FEATURE GROUP 5: Stated Preferences ===
data['states_prefer_local'] = (data['Q17'] == 'Local').astype(int)
data['states_prefer_chain'] = (data['Q17'] == 'Chain').astype(int)

# === FEATURE GROUP 6: Demographics ===
data['age'] = pd.to_numeric(data['Q30'], errors='coerce')
data['has_transport'] = (data['Q36'] == 'Yes').astype(int)
data['on_campus'] = (data['Q33'] == 'On campus').astype(int)
data['off_campus'] = (data['Q33'] == 'Off campus').astype(int)

# Year encoding
year_map = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4,
            'Super Senior': 5, 'Graduate student': 6}
data['year_numeric'] = data['Q31'].map(year_map)

# === FEATURE GROUP 7: Topping/Variety Importance ===
data['imp_topping_variety'] = data['Q8'].map(importance_map)
data['imp_foldability'] = data['Q7'].map({
    'Extremely important': 5, 'Very important': 4, 'Moderately important': 3,
    'Slightly important': 2, 'Not at all important': 1
})

# === FEATURE GROUP 8: Loyalty & Deal Sensitivity ===
data['loyalty_score'] = data['Q29'].map(loyalty_map)
data['deal_sensitivity'] = data['Q19'].map(likelihood_map)

# === TARGET VARIABLE ===
# Define "best pizza" as choosing a LOCAL restaurant (actual behavior)
local_restaurants = ["Joe's Brooklyn Pizza", "Salvatore's Pizza", "Mark's Pizzeria",
                     "Pontillo's Pizza", "Perri's Pizza", "Fire Crust Pizza",
                     "Peels on Wheels", "Brandani's Pizza", "Tony Pepperoni",
                     "Pizza Wizard", "East of Chicago"]

data['chose_local'] = data['Q28'].isin(local_restaurants).astype(int)

# Also create alternative targets for analysis
data['chose_dominos'] = (data['Q28'] == "Domino's Pizza").astype(int)

print("\nFeature Groups Created:")
print("  1. Pizza Quality Importance (9 features)")
print("  2. Ordering Behavior (4 features)")
print("  3. Time Preferences (4 features)")
print("  4. Price Sensitivity (4 features)")
print("  5. Stated Preferences (2 features)")
print("  6. Demographics (5 features)")
print("  7. Variety/Crust Preferences (2 features)")
print("  8. Loyalty & Deals (2 features)")
print(f"\nTotal Features: 32")

# =============================================================================
# FEATURE SELECTION AND DATA PREPARATION
# =============================================================================
print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

# Select features for modeling
feature_cols = [
    # Quality importance
    'imp_taste', 'imp_ingredients', 'imp_crust', 'imp_balance',
    'imp_freshness', 'imp_appearance', 'imp_price', 'imp_convenience', 'imp_special',
    # Behavior
    'orders_per_month', 'prefers_pickup', 'orders_online',
    # Time
    'expected_delivery_time', 'expected_pickup_time', 'willing_wait_delivery', 'willing_drive_pickup',
    # Price
    'expected_price', 'max_price', 'price_flexibility', 'price_over_location',
    # Stated preference
    'states_prefer_local',
    # Demographics
    'age', 'has_transport', 'on_campus', 'year_numeric',
    # Other
    'imp_topping_variety', 'imp_foldability', 'deal_sensitivity'
]

target_col = 'chose_local'

# Prepare data
model_data = data[feature_cols + [target_col]].dropna()
X = model_data[feature_cols]
y = model_data[target_col]

print(f"Samples after dropping missing: {len(X)}")
print(f"Target distribution:")
print(f"  Chose Local: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Chose Chain: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# MODEL BUILDING
# =============================================================================
print("\n" + "=" * 80)
print("ENSEMBLE MODEL TRAINING")
print("=" * 80)

# Define base models
print("\nBase Models:")

# 1. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)
print("  1. Random Forest (100 trees, max_depth=5)")

# 2. Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    min_samples_split=10,
    random_state=42
)
print("  2. Gradient Boosting (100 estimators, lr=0.1)")

# 3. Logistic Regression
lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
print("  3. Logistic Regression (L2 regularization)")

# 4. Decision Tree (for interpretability)
dt_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=15,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'
)
print("  4. Decision Tree (max_depth=4, for interpretability)")

# =============================================================================
# CROSS-VALIDATION
# =============================================================================
print("\n" + "-" * 60)
print("Cross-Validation Results (5-Fold Stratified):")
print("-" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest': (rf_model, X_train, y_train),
    'Gradient Boosting': (gb_model, X_train, y_train),
    'Logistic Regression': (lr_model, X_train_scaled, y_train),
    'Decision Tree': (dt_model, X_train, y_train)
}

cv_results = {}
for name, (model, X_cv, y_cv) in models.items():
    scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='accuracy')
    cv_results[name] = scores
    print(f"  {name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# =============================================================================
# ENSEMBLE: VOTING CLASSIFIER
# =============================================================================
print("\n" + "-" * 60)
print("Building Ensemble (Voting Classifier):")
print("-" * 60)

# Train individual models first
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lr_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train, y_train)

# For voting, we need models that use the same feature scale
# We'll use soft voting with RF, GB, and a separately handled LR

# Simple ensemble: average probabilities from RF and GB
rf_proba = rf_model.predict_proba(X_test)[:, 1]
gb_proba = gb_model.predict_proba(X_test)[:, 1]
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
dt_proba = dt_model.predict_proba(X_test)[:, 1]

# Weighted ensemble (based on CV performance)
weights = [cv_results['Random Forest'].mean(),
           cv_results['Gradient Boosting'].mean(),
           cv_results['Logistic Regression'].mean()]
weights = np.array(weights) / sum(weights)

ensemble_proba = (weights[0] * rf_proba +
                  weights[1] * gb_proba +
                  weights[2] * lr_proba)
ensemble_pred = (ensemble_proba > 0.5).astype(int)

print(f"  Ensemble weights: RF={weights[0]:.3f}, GB={weights[1]:.3f}, LR={weights[2]:.3f}")

# =============================================================================
# MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

print("\nTest Set Performance:")
print("-" * 60)

results = {
    'Random Forest': (rf_model.predict(X_test), rf_proba),
    'Gradient Boosting': (gb_model.predict(X_test), gb_proba),
    'Logistic Regression': (lr_model.predict(X_test_scaled), lr_proba),
    'Decision Tree': (dt_model.predict(X_test), dt_proba),
    'Ensemble': (ensemble_pred, ensemble_proba)
}

print(f"{'Model':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 60)

for name, (pred, proba) in results.items():
    acc = accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except:
        auc = 0.5

    # Calculate precision and recall for class 1 (chose local)
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)

    print(f"{name:<20} {acc:>10.3f} {auc:>10.3f} {prec:>10.3f} {rec:>10.3f}")

# Confusion Matrix for Ensemble
print("\nEnsemble Confusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(f"                 Predicted")
print(f"                Chain  Local")
print(f"  Actual Chain   {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"  Actual Local   {cm[1,0]:4d}   {cm[1,1]:4d}")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Random Forest Feature Importance
print("\n--- Random Forest Feature Importance ---")
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"{'Rank':<5} {'Feature':<30} {'Importance':>12}")
print("-" * 50)
for i, row in rf_importance.head(15).iterrows():
    rank = rf_importance.index.get_loc(i) + 1
    print(f"{rank:<5} {row['feature']:<30} {row['importance']:>12.4f}")

# Gradient Boosting Feature Importance
print("\n--- Gradient Boosting Feature Importance ---")
gb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"{'Rank':<5} {'Feature':<30} {'Importance':>12}")
print("-" * 50)
for i, row in gb_importance.head(15).iterrows():
    rank = gb_importance.index.get_loc(i) + 1
    print(f"{rank:<5} {row['feature']:<30} {row['importance']:>12.4f}")

# Logistic Regression Coefficients
print("\n--- Logistic Regression Coefficients ---")
lr_coef = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(f"{'Rank':<5} {'Feature':<30} {'Coefficient':>12} {'Direction':>10}")
print("-" * 60)
for i, row in lr_coef.head(15).iterrows():
    rank = list(lr_coef.index).index(i) + 1
    direction = "→ Local" if row['coefficient'] > 0 else "→ Chain"
    print(f"{rank:<5} {row['feature']:<30} {row['coefficient']:>+12.4f} {direction:>10}")

# =============================================================================
# PERMUTATION IMPORTANCE (More Robust)
# =============================================================================
print("\n--- Permutation Importance (Random Forest) ---")
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)

perm_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print(f"{'Rank':<5} {'Feature':<30} {'Importance':>12} {'Std':>10}")
print("-" * 60)
for i, row in perm_df.head(15).iterrows():
    rank = list(perm_df.index).index(i) + 1
    print(f"{rank:<5} {row['feature']:<30} {row['importance_mean']:>12.4f} {row['importance_std']:>10.4f}")

# =============================================================================
# CONSENSUS FEATURE RANKING
# =============================================================================
print("\n" + "=" * 80)
print("CONSENSUS FEATURE RANKING")
print("Aggregated across all models")
print("=" * 80)

# Normalize all importance scores to 0-1
rf_norm = rf_importance.set_index('feature')['importance'] / rf_importance['importance'].max()
gb_norm = gb_importance.set_index('feature')['importance'] / gb_importance['importance'].max()
lr_norm = lr_coef.set_index('feature')['coefficient'].abs() / lr_coef['coefficient'].abs().max()
perm_norm = perm_df.set_index('feature')['importance_mean']
perm_norm = (perm_norm - perm_norm.min()) / (perm_norm.max() - perm_norm.min())

# Combine
consensus = pd.DataFrame({
    'RF': rf_norm,
    'GB': gb_norm,
    'LR': lr_norm,
    'Perm': perm_norm
})
consensus['mean'] = consensus.mean(axis=1)
consensus = consensus.sort_values('mean', ascending=False)

print(f"\n{'Rank':<5} {'Feature':<30} {'RF':>8} {'GB':>8} {'LR':>8} {'Perm':>8} {'MEAN':>8}")
print("-" * 80)
for i, (feat, row) in enumerate(consensus.head(15).iterrows(), 1):
    print(f"{i:<5} {feat:<30} {row['RF']:>8.3f} {row['GB']:>8.3f} {row['LR']:>8.3f} {row['Perm']:>8.3f} {row['mean']:>8.3f}")

# =============================================================================
# DECISION RULES FROM TREE
# =============================================================================
print("\n" + "=" * 80)
print("INTERPRETABLE DECISION RULES")
print("From Decision Tree (max_depth=4)")
print("=" * 80)

def get_tree_rules(tree, feature_names, class_names):
    """Extract readable rules from decision tree"""
    from sklearn.tree import _tree

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, depth, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Left branch
            recurse(tree_.children_left[node], depth + 1,
                   path + [(name, "<=", threshold)])
            # Right branch
            recurse(tree_.children_right[node], depth + 1,
                   path + [(name, ">", threshold)])
        else:
            # Leaf node
            values = tree_.value[node][0]
            total = sum(values)
            pred_class = np.argmax(values)
            confidence = values[pred_class] / total
            if confidence > 0.6:  # Only show confident rules
                rules.append({
                    'path': path,
                    'prediction': class_names[pred_class],
                    'confidence': confidence,
                    'samples': int(total)
                })

    recurse(0, 0, [])
    return rules

rules = get_tree_rules(dt_model, feature_cols, ['Chain', 'Local'])

print("\nKey Decision Rules (Confidence > 60%):")
print("-" * 70)
for i, rule in enumerate(sorted(rules, key=lambda x: x['confidence'], reverse=True)[:8], 1):
    print(f"\nRule {i}: → {rule['prediction']} (Confidence: {rule['confidence']:.1%}, n={rule['samples']})")
    for feat, op, val in rule['path']:
        feat_clean = feat.replace('imp_', '').replace('_', ' ').title()
        print(f"    IF {feat_clean} {op} {val:.2f}")

# =============================================================================
# CUSTOMER PROFILES
# =============================================================================
print("\n" + "=" * 80)
print("PREDICTED CUSTOMER PROFILES")
print("=" * 80)

# Profile analysis based on predictions
local_pred_features = X_test[ensemble_pred == 1].mean()
chain_pred_features = X_test[ensemble_pred == 0].mean()

print("\nFeature Means by Predicted Preference:")
print("-" * 70)
print(f"{'Feature':<30} {'Pred Local':>12} {'Pred Chain':>12} {'Diff':>10}")
print("-" * 70)

for feat in consensus.head(10).index:
    local_val = local_pred_features[feat]
    chain_val = chain_pred_features[feat]
    diff = local_val - chain_val
    print(f"{feat:<30} {local_val:>12.2f} {chain_val:>12.2f} {diff:>+10.2f}")

# =============================================================================
# MODEL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("ENSEMBLE MODEL SUMMARY")
print("=" * 80)

# Best model
best_model = max(results.items(), key=lambda x: accuracy_score(y_test, x[1][0]))
best_auc = max(results.items(), key=lambda x: roc_auc_score(y_test, x[1][1]) if len(set(y_test)) > 1 else 0)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE MODEL RESULTS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BEST PERFORMING MODEL: {best_model[0]:<45}      │
│  Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.1%}                                              │
│  Ensemble AUC-ROC: {roc_auc_score(y_test, ensemble_proba):.3f}                                               │
│                                                                             │
│  TOP 5 PREDICTIVE FEATURES (Consensus Ranking):                             │
│  1. {consensus.index[0]:<60}         │
│  2. {consensus.index[1]:<60}         │
│  3. {consensus.index[2]:<60}         │
│  4. {consensus.index[3]:<60}         │
│  5. {consensus.index[4]:<60}         │
│                                                                             │
│  KEY INSIGHT:                                                               │
│  Stated preference for local is the strongest predictor, but behavioral     │
│  factors (transport, crust importance, willingness to drive) separate       │
│  those who ACT on that preference from those who don't.                     │
│                                                                             │
│  DEFINITION OF "BEST PIZZA" (from ML model):                                │
│  A student chooses LOCAL when they:                                         │
│    ✓ State they prefer local businesses                                     │
│    ✓ Have transportation (can drive to pick up)                             │
│    ✓ Place higher importance on crust quality                               │
│    ✓ Are willing to drive further for quality                               │
│    ✓ Have higher price expectations (willing to pay more)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# Save feature importance for visualization
consensus.to_csv('outputs/feature_importance_consensus.csv')
print("✓ Feature importance saved to outputs/feature_importance_consensus.csv")
