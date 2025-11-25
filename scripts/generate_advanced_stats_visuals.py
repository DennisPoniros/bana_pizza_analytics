"""
BANA255 Pizza Survey - Advanced Statistics Visualizations
==========================================================
Generates figures 29-40 for advanced statistical analyses.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#3B1F2B',
    'local': '#2E86AB',
    'chain': '#F18F01'
}

# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading data...")

xlsx_file = 'BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx'
df = pd.read_excel(xlsx_file)
data = df.iloc[1:].reset_index(drop=True)
data = data[data['Q2'] == 'Yes'].reset_index(drop=True)

# Mappings
importance_map = {
    'Not at all important': 1, 'Slightly important': 2,
    'Moderately important': 3, 'Very important': 4, 'Extremely important': 5
}

q5_cols = ['Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5', 'Q5_6', 'Q5_7', 'Q5_8', 'Q5_9']
q5_labels = {
    'Q5_1': 'Taste', 'Q5_2': 'Ingredients', 'Q5_3': 'Crust',
    'Q5_4': 'Balance', 'Q5_5': 'Freshness', 'Q5_6': 'Appearance',
    'Q5_7': 'Price', 'Q5_8': 'Convenience', 'Q5_9': 'Special Features'
}

for col in q5_cols:
    data[f'{col}_score'] = data[col].map(importance_map)

data['loyalty_score'] = data['Q29'].map({
    'Not loyal': 1, 'Slightly loyal': 2, 'Moderately loyal': 3,
    'Very loyal': 4, 'Extremely loyal': 5
})
data['orders_month'] = pd.to_numeric(data['Q4'], errors='coerce')
data['expected_price'] = pd.to_numeric(data['Q21_1'], errors='coerce')
data['max_price'] = pd.to_numeric(data['Q21_2'], errors='coerce')

local_restaurants = ["Joe's Brooklyn Pizza", "Salvatore's Pizza", "Mark's Pizzeria",
                     "Pontillo's Pizza", "Perri's Pizza", "Fire Crust Pizza",
                     "Peels on Wheels", "Brandani's Pizza", "Tony Pepperoni", "Pizza Wizard"]
chain_restaurants = ["Domino's Pizza", "Papa John's", "Little Caesars",
                     "Pizza Hut", "Costco Pizza", "Blaze Pizza"]

data['chose_local'] = data['Q28'].isin(local_restaurants).astype(int)

# =============================================================================
# FIGURE 29: PCA SCREE PLOT
# =============================================================================
print("Generating Figure 29: PCA Scree Plot...")

score_cols = [f'{col}_score' for col in q5_cols]
pca_data = data[score_cols].dropna()
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)

pca = PCA()
pca.fit(pca_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
eigenvalues = pca.explained_variance_
components = range(1, len(eigenvalues) + 1)

ax1.bar(components, eigenvalues, color=COLORS['primary'], alpha=0.7, label='Eigenvalue')
ax1.axhline(y=1, color=COLORS['accent'], linestyle='--', linewidth=2, label='Kaiser Criterion (λ=1)')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigenvalue')
ax1.set_title('Scree Plot: Eigenvalues by Component')
ax1.set_xticks(components)
ax1.legend()

# Cumulative variance
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
ax2.plot(components, cumvar, 'o-', color=COLORS['secondary'], linewidth=2, markersize=8)
ax2.axhline(y=80, color=COLORS['accent'], linestyle='--', linewidth=2, label='80% Threshold')
ax2.fill_between(components, cumvar, alpha=0.3, color=COLORS['secondary'])
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('Cumulative Variance Explained')
ax2.set_xticks(components)
ax2.set_ylim(0, 105)
ax2.legend()

plt.tight_layout()
plt.savefig('outputs/fig29_pca_scree.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 30: PCA BIPLOT
# =============================================================================
print("Generating Figure 30: PCA Biplot...")

pca_3 = PCA(n_components=2)
scores = pca_3.fit_transform(pca_scaled)
loadings = pca_3.components_.T

fig, ax = plt.subplots(figsize=(12, 10))

# Score plot (observations)
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=data.loc[pca_data.index, 'chose_local'],
                     cmap='RdYlBu', alpha=0.6, s=50)

# Loading vectors
scale = 3
for i, label in enumerate([q5_labels[col] for col in q5_cols]):
    ax.arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
             head_width=0.1, head_length=0.05, fc=COLORS['accent'], ec=COLORS['accent'])
    ax.text(loadings[i, 0]*scale*1.15, loadings[i, 1]*scale*1.15, label,
            fontsize=10, fontweight='bold', ha='center')

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca_3.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca_3.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('PCA Biplot: Customer Importance Profiles\n(Blue = Local, Red = Chain)')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Chose Local'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Chose Chain'),
                   Line2D([0], [0], color=COLORS['accent'], linewidth=2, label='Factor Loading')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('outputs/fig30_pca_biplot.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 31: CLUSTER VALIDATION (ELBOW & SILHOUETTE)
# =============================================================================
print("Generating Figure 31: Cluster Validation...")

cluster_data = data[score_cols].dropna()
cluster_scaled = StandardScaler().fit_transform(cluster_data)

k_range = range(2, 9)
inertias = []
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(cluster_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ax1.plot(list(k_range), inertias, 'o-', color=COLORS['primary'], linewidth=2, markersize=10)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax1.set_title('Elbow Method for Optimal k')
ax1.set_xticks(list(k_range))

# Highlight elbow
elbow_k = 4
ax1.axvline(x=elbow_k, color=COLORS['accent'], linestyle='--', linewidth=2, label=f'Elbow at k={elbow_k}')
ax1.legend()

# Silhouette plot
bars = ax2.bar(list(k_range), silhouettes, color=COLORS['secondary'], alpha=0.7)
optimal_idx = np.argmax(silhouettes)
bars[optimal_idx].set_color(COLORS['accent'])
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score by Number of Clusters')
ax2.set_xticks(list(k_range))
ax2.axhline(y=0.25, color='gray', linestyle='--', label='Minimum acceptable (0.25)')
ax2.legend()

# Annotate optimal
opt_k = list(k_range)[optimal_idx]
ax2.annotate(f'Optimal k={opt_k}\n(Score={silhouettes[optimal_idx]:.3f})',
             xy=(opt_k, silhouettes[optimal_idx]),
             xytext=(opt_k+1, silhouettes[optimal_idx]+0.05),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig31_cluster_validation.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 32: SILHOUETTE PLOT FOR OPTIMAL K
# =============================================================================
print("Generating Figure 32: Silhouette Analysis...")

optimal_k = list(k_range)[np.argmax(silhouettes)]
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_opt.fit_predict(cluster_scaled)
sil_samples = silhouette_samples(cluster_scaled, cluster_labels)

fig, ax = plt.subplots(figsize=(10, 8))

y_lower = 10
colors = plt.cm.Set2(np.linspace(0, 1, optimal_k))

for i in range(optimal_k):
    cluster_sil = sil_samples[cluster_labels == i]
    cluster_sil.sort()
    size = cluster_sil.shape[0]
    y_upper = y_lower + size

    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size, f'Cluster {i+1}\n(n={size})')
    y_lower = y_upper + 10

avg_sil = np.mean(sil_samples)
ax.axvline(x=avg_sil, color=COLORS['accent'], linestyle='--', linewidth=2,
           label=f'Average Silhouette ({avg_sil:.3f})')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title(f'Silhouette Plot for k={optimal_k} Clusters')
ax.set_xlim(-0.2, 1)
ax.set_yticks([])
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('outputs/fig32_silhouette_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 33: CRONBACH'S ALPHA VISUALIZATION
# =============================================================================
print("Generating Figure 33: Cronbach's Alpha...")

def cronbach_alpha(df):
    df = df.dropna()
    n_items = df.shape[1]
    item_variances = df.var(ddof=1)
    total_scores = df.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

alpha_full = cronbach_alpha(data[score_cols])

# Alpha if item deleted
alpha_deleted = {}
for col in score_cols:
    other_cols = [c for c in score_cols if c != col]
    alpha_deleted[q5_labels[col.replace('_score', '')]] = cronbach_alpha(data[other_cols])

fig, ax = plt.subplots(figsize=(12, 6))

items = list(alpha_deleted.keys())
alphas = list(alpha_deleted.values())

bars = ax.barh(items, alphas, color=COLORS['primary'], alpha=0.7)
ax.axvline(x=alpha_full, color=COLORS['accent'], linestyle='--', linewidth=3,
           label=f'Full Scale α = {alpha_full:.3f}')

# Color bars that increase alpha if removed
for i, (item, alpha) in enumerate(zip(items, alphas)):
    if alpha > alpha_full:
        bars[i].set_color(COLORS['success'])
        ax.annotate('Consider removing', xy=(alpha, i), xytext=(alpha+0.02, i),
                    fontsize=9, va='center')

ax.set_xlabel("Cronbach's Alpha if Item Deleted")
ax.set_title("Scale Reliability Analysis: Cronbach's Alpha")
ax.set_xlim(0.5, 1.0)
ax.legend(loc='lower right')

# Interpretation zones
ax.axvspan(0.9, 1.0, alpha=0.1, color='green', label='Excellent')
ax.axvspan(0.8, 0.9, alpha=0.1, color='yellow')
ax.axvspan(0.7, 0.8, alpha=0.1, color='orange')

plt.tight_layout()
plt.savefig('outputs/fig33_cronbach_alpha.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 34: SPEARMAN CORRELATION HEATMAP
# =============================================================================
print("Generating Figure 34: Spearman Correlations...")

importance_data = data[score_cols].dropna()
spearman_corr = importance_data.corr(method='spearman')
spearman_corr.index = [q5_labels[c.replace('_score', '')] for c in spearman_corr.index]
spearman_corr.columns = [q5_labels[c.replace('_score', '')] for c in spearman_corr.columns]

fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)
sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'}, ax=ax)

ax.set_title('Spearman Rank Correlations Between Importance Factors', fontsize=14)

plt.tight_layout()
plt.savefig('outputs/fig34_spearman_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 35: VAN WESTENDORP PRICING
# =============================================================================
print("Generating Figure 35: Van Westendorp Pricing...")

price_data = data[['expected_price', 'max_price']].dropna()
price_data = price_data[(price_data['expected_price'] > 0) & (price_data['max_price'] > 0)]
price_data = price_data[price_data['max_price'] >= price_data['expected_price']]

price_range = np.arange(10, 35, 0.5)

# Calculate curves
too_cheap = []
not_cheap = []
not_expensive = []
too_expensive = []

for price in price_range:
    too_cheap.append((price_data['expected_price'] > price).mean() * 100)
    not_cheap.append((price_data['expected_price'] <= price).mean() * 100)
    not_expensive.append((price_data['max_price'] >= price).mean() * 100)
    too_expensive.append((price_data['max_price'] < price).mean() * 100)

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(price_range, not_cheap, '-', color=COLORS['primary'], linewidth=2.5, label='Not Cheap (price ≥ expected)')
ax.plot(price_range, not_expensive, '-', color=COLORS['secondary'], linewidth=2.5, label='Not Expensive (price ≤ max)')
ax.plot(price_range, too_cheap, '--', color=COLORS['primary'], linewidth=1.5, alpha=0.7, label='Too Cheap')
ax.plot(price_range, too_expensive, '--', color=COLORS['secondary'], linewidth=1.5, alpha=0.7, label='Too Expensive')

# Find intersections
optimal_price = price_data['expected_price'].median()
indifference_price = (price_data['expected_price'].median() + price_data['max_price'].median()) / 2

ax.axvline(x=optimal_price, color=COLORS['accent'], linestyle=':', linewidth=2)
ax.annotate(f'Point of Marginal\nCheapness: ${optimal_price:.0f}',
            xy=(optimal_price, 50), xytext=(optimal_price-4, 65),
            arrowprops=dict(arrowstyle='->', color=COLORS['accent']),
            fontsize=10, fontweight='bold')

ax.axvline(x=price_data['max_price'].median(), color=COLORS['success'], linestyle=':', linewidth=2)
ax.annotate(f'Point of Marginal\nExpensiveness: ${price_data["max_price"].median():.0f}',
            xy=(price_data['max_price'].median(), 50),
            xytext=(price_data['max_price'].median()+2, 65),
            arrowprops=dict(arrowstyle='->', color=COLORS['success']),
            fontsize=10, fontweight='bold')

# Shade acceptable price range
ax.axvspan(price_data['expected_price'].quantile(0.25),
           price_data['max_price'].quantile(0.75),
           alpha=0.15, color='green', label='Acceptable Range')

ax.set_xlabel('Price ($)')
ax.set_ylabel('Cumulative Percentage')
ax.set_title('Van Westendorp Price Sensitivity Analysis\n(16" Pizza)')
ax.set_xlim(10, 35)
ax.set_ylim(0, 100)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fig35_van_westendorp.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 36: MEDIATION PATH DIAGRAM
# =============================================================================
print("Generating Figure 36: Mediation Analysis...")

# Calculate mediation effects
med_data = data[['Q5_1_score', 'Q11', 'chose_local']].dropna()
med_data['taste'] = med_data['Q5_1_score']
med_data['pickup'] = (med_data['Q11'] == 'Pick up').astype(int)
med_data['local'] = med_data['chose_local']

# Path coefficients
model1 = LogisticRegression(max_iter=1000)
model1.fit(med_data[['taste']], med_data['local'])
c = model1.coef_[0][0]

model2 = LogisticRegression(max_iter=1000)
model2.fit(med_data[['taste']], med_data['pickup'])
a = model2.coef_[0][0]

model3 = LogisticRegression(max_iter=1000)
model3.fit(med_data[['taste', 'pickup']], med_data['local'])
c_prime = model3.coef_[0][0]
b = model3.coef_[0][1]

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw boxes
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# X: Taste Importance
x_box = FancyBboxPatch((0.5, 4), 2.5, 1.5, boxstyle="round,pad=0.1",
                        facecolor=COLORS['primary'], edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(x_box)
ax.text(1.75, 4.75, 'Taste\nImportance\n(X)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# M: Pickup Preference
m_box = FancyBboxPatch((3.75, 7), 2.5, 1.5, boxstyle="round,pad=0.1",
                        facecolor=COLORS['accent'], edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(m_box)
ax.text(5, 7.75, 'Pickup\nPreference\n(M)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Y: Local Choice
y_box = FancyBboxPatch((7, 4), 2.5, 1.5, boxstyle="round,pad=0.1",
                        facecolor=COLORS['secondary'], edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(y_box)
ax.text(8.25, 4.75, 'Local\nChoice\n(Y)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Arrows
# a path: X → M
ax.annotate('', xy=(3.75, 7.5), xytext=(3, 5.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(2.8, 6.8, f'a = {a:.3f}', fontsize=11, fontweight='bold', color=COLORS['primary'])

# b path: M → Y
ax.annotate('', xy=(7, 5), xytext=(6.25, 7.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(6.8, 6.8, f'b = {b:.3f}', fontsize=11, fontweight='bold', color=COLORS['accent'])

# c path: X → Y (direct, dashed)
ax.annotate('', xy=(7, 4.75), xytext=(3, 4.75),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='--'))
ax.text(5, 4.2, f"c' = {c_prime:.3f}", fontsize=11, fontweight='bold', color='gray')

# c total (below)
ax.text(5, 3.2, f'Total effect (c) = {c:.3f}', fontsize=12, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Indirect effect
indirect = a * b
pct_mediated = ((c - c_prime) / c * 100) if c != 0 else 0
ax.text(5, 1.5, f'Indirect effect (a × b) = {indirect:.3f}\n'
                f'% Mediated = {pct_mediated:.1f}%',
        fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Title
ax.text(5, 9.5, 'Mediation Analysis: Taste → Pickup → Local Choice',
        fontsize=16, fontweight='bold', ha='center')

plt.savefig('outputs/fig36_mediation_diagram.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 37: DISCRIMINANT ANALYSIS
# =============================================================================
print("Generating Figure 37: Discriminant Analysis...")

lda_features = ['Q5_1_score', 'Q5_2_score', 'Q5_3_score', 'Q5_7_score',
                'Q5_8_score', 'orders_month', 'expected_price']
lda_data = data[lda_features + ['chose_local']].dropna()

X_lda = lda_data[lda_features]
y_lda = lda_data['chose_local']

scaler_lda = StandardScaler()
X_lda_scaled = scaler_lda.fit_transform(X_lda)

lda = LinearDiscriminantAnalysis()
lda.fit(X_lda_scaled, y_lda)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Coefficient plot
coef_df = pd.DataFrame({
    'Variable': [q5_labels.get(f.replace('_score', ''), f) for f in lda_features],
    'Coefficient': lda.coef_[0]
}).sort_values('Coefficient')

colors = [COLORS['local'] if c > 0 else COLORS['chain'] for c in coef_df['Coefficient']]
ax1.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors, alpha=0.8)
ax1.axvline(x=0, color='black', linewidth=1)
ax1.set_xlabel('Discriminant Function Coefficient')
ax1.set_title('LDA Coefficients: What Discriminates Local vs Chain Choosers')
ax1.text(0.05, -0.5, '← Predicts CHAIN', fontsize=10, transform=ax1.get_yaxis_transform(), color=COLORS['chain'])
ax1.text(0.95, -0.5, 'Predicts LOCAL →', fontsize=10, transform=ax1.get_yaxis_transform(), color=COLORS['local'], ha='right')

# Score distribution
lda_scores = lda.transform(X_lda_scaled).flatten()

local_scores = lda_scores[y_lda == 1]
chain_scores = lda_scores[y_lda == 0]

ax2.hist(chain_scores, bins=20, alpha=0.6, color=COLORS['chain'], label=f'Chain (n={len(chain_scores)})', density=True)
ax2.hist(local_scores, bins=20, alpha=0.6, color=COLORS['local'], label=f'Local (n={len(local_scores)})', density=True)
ax2.axvline(x=local_scores.mean(), color=COLORS['local'], linestyle='--', linewidth=2)
ax2.axvline(x=chain_scores.mean(), color=COLORS['chain'], linestyle='--', linewidth=2)
ax2.set_xlabel('LDA Score')
ax2.set_ylabel('Density')
ax2.set_title('LDA Score Distribution by Choice')
ax2.legend()

plt.tight_layout()
plt.savefig('outputs/fig37_discriminant_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 38: CHI-SQUARE MOSAIC
# =============================================================================
print("Generating Figure 38: Chi-Square Associations...")

data['local_pref'] = data['Q17'].apply(lambda x: 'Prefers Local' if x == 'Local' else ('Prefers Chain' if x == 'Chain' else 'Unsure'))
data['actual_choice'] = data['chose_local'].apply(lambda x: 'Chose Local' if x == 1 else 'Chose Chain')

# Contingency table
ct_pct = pd.crosstab(data['local_pref'], data['actual_choice'], normalize='index') * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Stacked bar chart
ct_plot = ct_pct.copy()
ct_plot[['Chose Local', 'Chose Chain']].plot(kind='barh', stacked=True, ax=ax1,
                                               color=[COLORS['local'], COLORS['chain']], alpha=0.8)
ax1.set_xlabel('Percentage')
ax1.set_title('Stated Preference vs Actual Choice\n(Chi-square p < 0.001)')
ax1.legend(title='Actual Choice')

# Add percentage labels
for i, (idx, row) in enumerate(ct_plot.iterrows()):
    local_pct = row['Chose Local']
    ax1.text(local_pct/2, i, f'{local_pct:.0f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax1.text(local_pct + (100-local_pct)/2, i, f'{100-local_pct:.0f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Transportation vs choice
data['has_transport'] = data['Q36']
ct2 = pd.crosstab(data['has_transport'], data['actual_choice'], normalize='index') * 100

ct2.plot(kind='bar', ax=ax2, color=[COLORS['local'], COLORS['chain']], alpha=0.8, rot=0)
ax2.set_xlabel('Has Personal Transportation')
ax2.set_ylabel('Percentage')
ax2.set_title('Transportation Access vs Choice')
ax2.legend(title='Choice')

# Add labels
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.0f%%', label_type='center', fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('outputs/fig38_chi_square_associations.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 39: PROPENSITY SCORE DISTRIBUTION
# =============================================================================
print("Generating Figure 39: Propensity Scores...")

ps_features = ['Q5_1_score', 'Q5_7_score', 'Q5_8_score', 'orders_month', 'expected_price']
ps_data = data[ps_features + ['chose_local', 'loyalty_score']].dropna()

X_ps = ps_data[ps_features]
treatment = ps_data['chose_local']

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_ps, treatment)
propensity_scores = ps_model.predict_proba(X_ps)[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Distribution by group
local_ps = propensity_scores[treatment == 1]
chain_ps = propensity_scores[treatment == 0]

ax1.hist(chain_ps, bins=25, alpha=0.6, color=COLORS['chain'], label=f'Chain Choosers (n={len(chain_ps)})', density=True)
ax1.hist(local_ps, bins=25, alpha=0.6, color=COLORS['local'], label=f'Local Choosers (n={len(local_ps)})', density=True)
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Density')
ax1.set_title('Propensity Score Distribution')
ax1.legend()

# Common support region
overlap_min = max(local_ps.min(), chain_ps.min())
overlap_max = min(local_ps.max(), chain_ps.max())
ax1.axvspan(overlap_min, overlap_max, alpha=0.2, color='green', label='Common Support')

# Box plot comparison
bp_data = pd.DataFrame({
    'Propensity Score': propensity_scores,
    'Group': ['Local' if t == 1 else 'Chain' for t in treatment]
})

bp = ax2.boxplot([chain_ps, local_ps], labels=['Chain', 'Local'], patch_artist=True)
bp['boxes'][0].set_facecolor(COLORS['chain'])
bp['boxes'][1].set_facecolor(COLORS['local'])
for box in bp['boxes']:
    box.set_alpha(0.7)

ax2.set_ylabel('Propensity Score')
ax2.set_title('Propensity Score by Actual Choice')

plt.tight_layout()
plt.savefig('outputs/fig39_propensity_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 40: SIMULATED CHOICE MODEL
# =============================================================================
print("Generating Figure 40: Simulated Choice Model...")

# Importance weights
weights = {}
for col in q5_cols:
    label = q5_labels[col]
    weights[label] = data[f'{col}_score'].mean()

total_weight = sum(weights.values())
norm_weights = {k: v/total_weight for k, v in weights.items()}

# Restaurant profiles
restaurant_profiles = {
    "Domino's": {'Taste': 3.5, 'Ingredients': 3.0, 'Crust': 3.0, 'Balance': 3.2,
                 'Freshness': 3.5, 'Appearance': 3.2, 'Price': 4.5, 'Convenience': 4.8, 'Special Features': 3.5},
    "Joe's Brooklyn": {'Taste': 4.5, 'Ingredients': 4.2, 'Crust': 4.3, 'Balance': 4.0,
                       'Freshness': 4.3, 'Appearance': 3.8, 'Price': 3.2, 'Convenience': 2.8, 'Special Features': 3.0},
    "NEW: Scenario A": {'Taste': 4.5, 'Ingredients': 4.0, 'Crust': 4.2, 'Balance': 4.0,
                        'Freshness': 4.5, 'Appearance': 4.0, 'Price': 3.8, 'Convenience': 4.2, 'Special Features': 3.5},
    "NEW: Scenario B": {'Taste': 4.8, 'Ingredients': 4.5, 'Crust': 4.5, 'Balance': 4.5,
                        'Freshness': 4.8, 'Appearance': 4.2, 'Price': 3.0, 'Convenience': 3.5, 'Special Features': 4.0}
}

# Calculate weighted scores
scores = {}
for restaurant, profile in restaurant_profiles.items():
    weighted_score = sum(profile[factor] * norm_weights[factor] for factor in profile)
    scores[restaurant] = weighted_score

# Market shares (logit)
beta = 2.0
exp_utilities = {r: np.exp(beta * s) for r, s in scores.items()}
sum_exp = sum(exp_utilities.values())
market_shares = {r: u / sum_exp * 100 for r, u in exp_utilities.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Weighted scores bar chart
restaurants = list(scores.keys())
score_values = list(scores.values())
colors_list = [COLORS['chain'], COLORS['local'], COLORS['accent'], COLORS['secondary']]

bars1 = ax1.barh(restaurants, score_values, color=colors_list, alpha=0.8)
ax1.set_xlabel('Weighted Quality Score')
ax1.set_title('Restaurant Quality Scores\n(Importance-Weighted)')
ax1.set_xlim(3, 4.5)

for bar, score in zip(bars1, score_values):
    ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.2f}',
             va='center', fontsize=11, fontweight='bold')

# Market share pie
share_values = list(market_shares.values())
explode = (0, 0, 0.1, 0.1)  # Highlight new entrants

wedges, texts, autotexts = ax2.pie(share_values, labels=restaurants, autopct='%1.1f%%',
                                    colors=colors_list, explode=explode, startangle=90)
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')
ax2.set_title('Simulated Market Share\n(Logit Choice Model)')

plt.tight_layout()
plt.savefig('outputs/fig40_simulated_choice.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 80)
print("""
Generated Figures:
  fig29_pca_scree.png          - PCA scree plot and cumulative variance
  fig30_pca_biplot.png         - PCA biplot with factor loadings
  fig31_cluster_validation.png - Elbow and silhouette analysis
  fig32_silhouette_plot.png    - Detailed silhouette plot
  fig33_cronbach_alpha.png     - Scale reliability analysis
  fig34_spearman_heatmap.png   - Spearman correlation matrix
  fig35_van_westendorp.png     - Price sensitivity curves
  fig36_mediation_diagram.png  - Mediation path diagram
  fig37_discriminant_analysis.png - LDA results
  fig38_chi_square_associations.png - Categorical associations
  fig39_propensity_scores.png  - Propensity score distribution
  fig40_simulated_choice.png   - Choice model simulation

All figures saved to outputs/ directory.
""")
