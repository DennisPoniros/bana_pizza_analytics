"""
BANA255 Pizza Survey - Machine Learning Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = Path("outputs")

# =============================================================================
# FIGURE 13: Feature Importance Consensus
# =============================================================================
print("Generating ML visualizations...")

# Load consensus data
try:
    consensus = pd.read_csv(OUTPUT_DIR / 'feature_importance_consensus.csv', index_col=0)
except:
    # Recreate from model output
    consensus = pd.DataFrame({
        'feature': ['states_prefer_local', 'expected_pickup_time', 'prefers_pickup',
                   'max_price', 'price_flexibility', 'expected_price', 'orders_online',
                   'willing_drive_pickup', 'age', 'orders_per_month'],
        'mean': [0.778, 0.625, 0.576, 0.522, 0.512, 0.436, 0.423, 0.418, 0.405, 0.405]
    }).set_index('feature')

fig, ax = plt.subplots(figsize=(10, 8))

# Clean feature names
feature_names = {
    'states_prefer_local': 'States "Prefer Local"',
    'expected_pickup_time': 'Expected Pickup Time',
    'prefers_pickup': 'Prefers Pickup',
    'max_price': 'Max Price Willing to Pay',
    'price_flexibility': 'Price Flexibility',
    'expected_price': 'Expected Price',
    'orders_online': 'Orders Online',
    'willing_drive_pickup': 'Willing to Drive (Pickup)',
    'age': 'Age',
    'orders_per_month': 'Orders per Month',
    'imp_foldability': 'Foldability Importance',
    'imp_crust': 'Crust Importance',
    'expected_delivery_time': 'Expected Delivery Time',
    'price_over_location': 'Price Over Location',
    'imp_price': 'Price Importance'
}

top_features = consensus.head(15).copy()
top_features['clean_name'] = [feature_names.get(f, f) for f in top_features.index]

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(top_features)))[::-1]

bars = ax.barh(top_features['clean_name'][::-1], top_features['mean'].values[::-1], color=colors)
ax.set_xlabel('Consensus Importance Score (0-1)')
ax.set_title('Top 15 Features Predicting Local vs Chain Choice\n(Consensus across RF, GB, LR, Permutation)',
             fontsize=12, fontweight='bold')

for bar, val in zip(bars, top_features['mean'].values[::-1]):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig13_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig13_feature_importance.png")

# =============================================================================
# FIGURE 14: Model Performance Comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Model accuracy comparison
models = ['Random\nForest', 'Gradient\nBoosting', 'Logistic\nRegression', 'Decision\nTree', 'Ensemble']
accuracy = [0.711, 0.684, 0.632, 0.632, 0.605]
auc_roc = [0.756, 0.643, 0.685, 0.804, 0.708]

x = np.arange(len(models))
width = 0.35

bars1 = axes[0].bar(x - width/2, accuracy, width, label='Accuracy', color='#3498db')
bars2 = axes[0].bar(x + width/2, auc_roc, width, label='AUC-ROC', color='#2ecc71')

axes[0].set_ylabel('Score')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].set_ylim(0, 1)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
axes[0].set_title('Model Performance Comparison', fontsize=11, fontweight='bold')

for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)

# Cross-validation scores
cv_means = [0.618, 0.601, 0.591, 0.458]
cv_stds = [0.090, 0.066, 0.122, 0.106]
cv_models = ['Random\nForest', 'Gradient\nBoosting', 'Logistic\nReg', 'Decision\nTree']

bars = axes[1].bar(cv_models, cv_means, yerr=cv_stds, capsize=5,
                   color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
axes[1].set_ylabel('Cross-Validation Accuracy')
axes[1].set_ylim(0, 1)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('5-Fold Cross-Validation\n(with standard deviation)', fontsize=11, fontweight='bold')

for bar, mean in zip(bars, cv_means):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig14_model_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig14_model_performance.png")

# =============================================================================
# FIGURE 15: Decision Rules Visualization
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Create a visual decision tree representation
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw boxes and arrows
def draw_box(x, y, text, color, width=2.5, height=0.8):
    rect = plt.Rectangle((x-width/2, y-height/2), width, height,
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_arrow(x1, y1, x2, y2, text=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    if text:
        ax.text(mid_x, mid_y + 0.2, text, fontsize=8, ha='center')

# Root node
draw_box(5, 9, 'States "Prefer Local"?', '#f0f0f0', width=3)

# Level 1
draw_arrow(5, 8.6, 2.5, 7.8, 'No')
draw_arrow(5, 8.6, 7.5, 7.8, 'Yes')

draw_box(2.5, 7.4, '→ CHAIN\n(69% of "No")', '#e74c3c', width=2.2)
draw_box(7.5, 7.4, 'Prefers Pickup?', '#f0f0f0', width=2.5)

# Level 2 - right branch
draw_arrow(7.5, 7, 6, 6.2, 'No')
draw_arrow(7.5, 7, 9, 6.2, 'Yes')

draw_box(6, 5.8, '→ CHAIN\n(69%)', '#e74c3c', width=1.8)
draw_box(9, 5.8, 'Orders Online?', '#f0f0f0', width=2.2)

# Level 3
draw_arrow(9, 5.4, 8, 4.6, 'No')
draw_arrow(9, 5.4, 10, 4.6, 'Yes')

draw_box(8, 4.2, '→ LOCAL\n(83%)', '#2ecc71', width=1.8)
draw_box(10, 4.2, 'Exp. Delivery\n≤35 min?', '#f0f0f0', width=1.8, height=1)

# Level 4
draw_arrow(10, 3.7, 9.3, 2.8, 'Yes')
draw_arrow(10, 3.7, 10.7, 2.8, 'No')

draw_box(9.3, 2.4, '→ LOCAL\n(75%)', '#2ecc71', width=1.6)
draw_box(10.7, 2.4, '→ CHAIN\n(65%)', '#e74c3c', width=1.6)

ax.set_title('Decision Tree Rules: Who Chooses Local vs Chain?\n(Simplified from max_depth=4 tree)',
             fontsize=12, fontweight='bold', pad=20)

# Add legend
local_patch = mpatches.Patch(color='#2ecc71', label='Predicts LOCAL')
chain_patch = mpatches.Patch(color='#e74c3c', label='Predicts CHAIN')
ax.legend(handles=[local_patch, chain_patch], loc='lower left')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig15_decision_rules.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig15_decision_rules.png")

# =============================================================================
# FIGURE 16: Customer Profile Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Profile data from model
features = ['States Prefer Local', 'Prefers Pickup', 'Expected Price ($)',
            'Max Price ($)', 'Orders/Month', 'Willing to Drive (min)']
local_vals = [0.73, 1.00, 20.80, 27.00, 2.67, 33.13]
chain_vals = [0.43, 0.43, 16.78, 24.78, 1.58, 28.57]

# Normalize for radar-like comparison
local_norm = [0.73, 1.00, 20.80/30, 27.00/35, 2.67/5, 33.13/50]
chain_norm = [0.43, 0.43, 16.78/30, 24.78/35, 1.58/5, 28.57/50]

x = np.arange(len(features))
width = 0.35

bars1 = ax.barh(x - width/2, local_norm, width, label='Predicted LOCAL', color='#2ecc71')
bars2 = ax.barh(x + width/2, chain_norm, width, label='Predicted CHAIN', color='#e74c3c')

ax.set_yticks(x)
ax.set_yticklabels(features)
ax.set_xlabel('Normalized Value (0-1)')
ax.set_title('Customer Profile: Local vs Chain Choosers\n(Based on Ensemble Model Predictions)',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right')

# Add actual values
for i, (bar, val) in enumerate(zip(bars1, local_vals)):
    if val > 1:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}', va='center', fontsize=8, color='#2ecc71')
    else:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.0%}', va='center', fontsize=8, color='#2ecc71')

for i, (bar, val) in enumerate(zip(bars2, chain_vals)):
    if val > 1:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}', va='center', fontsize=8, color='#e74c3c')
    else:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.0%}', va='center', fontsize=8, color='#e74c3c')

ax.set_xlim(0, 1.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig16_customer_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig16_customer_profiles.png")

# =============================================================================
# FIGURE 17: Feature Category Importance
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Stated\nPreference', 'Pickup\nBehavior', 'Price\nFactors',
              'Time\nExpectations', 'Demographics', 'Quality\nImportance']
importance = [0.778, 0.576, 0.490, 0.457, 0.405, 0.380]

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))

bars = ax.bar(categories, importance, color=colors)
ax.set_ylabel('Mean Consensus Importance')
ax.set_title('Feature Category Importance\n(What matters most in predicting pizza preference?)',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)

for bar, val in zip(bars, importance):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
           f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig17_category_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: fig17_category_importance.png")

print("\n✓ All ML visualizations generated!")
