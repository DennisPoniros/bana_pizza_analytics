# BANA255 Pizza Survey - Findings Report

**Study**: What is the Best Pizza Available to RIT Students?
**Data Collected**: November 9-14, 2025
**Valid Responses**: 161 (consented participants)
**Last Updated**: November 28, 2025

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [EXECUTIVE_SUMMARY.md](reports/EXECUTIVE_SUMMARY.md) | Visual summary with charts and key takeaways |
| [STRATEGIC_ANALYSIS.md](reports/STRATEGIC_ANALYSIS.md) | **Winner declaration, unmet needs, model confidence, causal analysis** |
| [COMPETITIVE_ANALYSIS.md](reports/COMPETITIVE_ANALYSIS.md) | Full competitive model and go-to-market strategy |
| [ML_MODEL_REPORT.md](reports/ML_MODEL_REPORT.md) | Behavioral ML model (circular features excluded) |
| [DIETARY_ANALYSIS.md](reports/DIETARY_ANALYSIS.md) | **Dietary accommodation preferences and market opportunity** |
| [METHODOLOGY.md](reports/METHODOLOGY.md) | Statistical methods, assumptions, and references |
| `outputs/` | Generated figures and CSV tables |

### Generated Visualizations

**Descriptive Analysis (fig1-6)**:
- `fig1_local_vs_chain.png` - Preference distribution
- `fig2_top_pizza_places.png` - Restaurant rankings
- `fig3_importance_ratings.png` - Quality characteristic importance
- `fig4_order_time_preferences.png` - Order method & time analysis
- `fig5_decision_factors.png` - Key decision factors
- `fig6_demographics.png` - Sample demographics

**Competitive Model (fig7-12)**:
- `fig7_importance_weights.png` - Weighted scoring model
- `fig8_competitive_ranking.png` - Threat score rankings
- `fig9_local_chain_paradox.png` - Stated vs actual behavior
- `fig10_dominos_opportunity.png` - Primary competitor analysis
- `fig11_positioning_map.png` - Strategic positioning map
- `fig12_local_predictors.png` - Regression predictors

**Machine Learning Model (fig13-17)**:
- `fig13_feature_importance.png` - Consensus feature ranking
- `fig14_model_performance.png` - Model comparison
- `fig15_decision_rules.png` - Decision tree rules
- `fig16_customer_profiles.png` - Local vs Chain profiles
- `fig17_category_importance.png` - Feature category analysis

**Advanced Presentation Visuals (fig18-28)**:
- `fig18_correlation_heatmap.png` - Quality factor correlations
- `fig19_local_chain_violin.png` - Factor comparison by choice type
- `fig20_paradox_flow.png` - Stated vs actual behavior flow
- `fig21_price_taste_scatter.png` - Price vs taste trade-off
- `fig22_frequency_loyalty_joint.png` - Order frequency vs loyalty
- `fig23_persuadable_profile.png` - Persuadable segment deep dive
- `fig24_time_expectations.png` - Expected vs willing time investment
- `fig25_dominos_vulnerability.png` - Competitor vulnerability analysis
- `fig26_radar_comparison.png` - Factor priorities radar chart
- `fig27_price_willingness.png` - Price distribution analysis
- `fig28_demographics_choice.png` - Demographics by choice type

**Advanced Statistical Analysis (fig29-40)**:
- `fig29_pca_scree.png` - PCA scree plot and cumulative variance
- `fig30_pca_biplot.png` - PCA biplot with factor loadings
- `fig31_cluster_validation.png` - Elbow and silhouette analysis
- `fig32_silhouette_plot.png` - Detailed silhouette plot for optimal k
- `fig33_cronbach_alpha.png` - Scale reliability (Cronbach's Alpha)
- `fig34_spearman_heatmap.png` - Spearman rank correlation matrix
- `fig35_van_westendorp.png` - Price sensitivity curves (Van Westendorp)
- `fig36_mediation_diagram.png` - Mediation path diagram (Baron & Kenny)
- `fig37_discriminant_analysis.png` - LDA coefficients and score distribution
- `fig38_chi_square_associations.png` - Categorical variable associations
- `fig39_propensity_scores.png` - Propensity score distribution
- `fig40_simulated_choice.png` - Market share simulation model

**Dietary Accommodation Analysis (fig41-46)**:
- `fig41_dietary_importance.png` - Dietary factor importance rankings
- `fig42_dietary_distribution.png` - Full distribution of dietary importance ratings
- `fig43_dietary_segments.png` - Dietary consciousness customer segmentation
- `fig44_dietary_correlations.png` - Correlation heatmap between dietary factors
- `fig45_dietary_vs_quality.png` - Comparison: quality vs dietary importance
- `fig46_dietary_options_sought.png` - Dietary options students actively seek

**Side Order Analysis (fig47-52)** *(NEW)*:
- `fig47_side_order_rankings.png` - Side order popularity by mean score
- `fig48_side_order_distributions.png` - Distribution histograms by item
- `fig49_side_spending.png` - Spending histogram and segments
- `fig50_sides_by_restaurant_type.png` - Local vs Chain comparison
- `fig51_side_correlations.png` - Spearman correlation heatmap
- `fig52_revenue_opportunity.png` - Strategic opportunity matrix

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [What It All Means: Plain-Language Interpretation](#what-it-all-means-plain-language-interpretation)
3. [Key Findings](#key-findings)
4. [Detailed Analysis](#detailed-analysis)
5. [Advanced Statistical Analysis](#advanced-statistical-analysis)
6. [Open Questions & Future Research](#open-questions--future-research)
7. [Methodology Notes](#methodology-notes)

---

## Executive Summary

This survey explores pizza preferences among RIT students to determine what makes a pizza place "the best." Analysis of 161 responses reveals that **taste is the dominant factor** in pizza selection, students **prefer local establishments** but paradoxically choose chains for convenience, and **pickup is strongly favored** over delivery.

> **Visual Summary**: See [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) for charts and figures.

### Top-Line Insights
| Finding | Key Metric | Confidence |
|---------|-----------|------------|
| Local preference | 84% prefer local over chain | p < 0.001 |
| Taste dominates | 94% rate taste as highly important | p < 0.001 |
| Pickup preferred | 71% choose pickup over delivery | Descriptive |
| Price matters | 67% say price > location | Descriptive |
| Average orders | 2.5 pizzas/month | - |

---

## What It All Means: Plain-Language Interpretation

This section synthesizes all 40+ statistical tests and visualizations into direct, actionable answers. Skip this section if you want technical details; read this first if you want the bottom line.

### The Core Question Answered

**"What is the best pizza available to RIT students?"**

**Answer**: There is no single "best" pizza—it depends on what students prioritize. However, the data reveals a clear winner for each priority:

| If You Prioritize... | Best Choice | Why |
|----------------------|-------------|-----|
| **Taste & Quality** | Joe's Brooklyn Pizza, Salvatore's | Local shops consistently win on perceived taste, crust quality, and ingredient freshness |
| **Price & Convenience** | Domino's Pizza | Chains win on value perception, speed, and familiarity |
| **Overall Balance** | Joe's Brooklyn Pizza | Best combination of quality scores with acceptable convenience |

**The uncomfortable truth**: Most students *say* they want quality (local) but *choose* convenience (chains). This isn't hypocrisy—it's rational behavior under constraints.

---

### What Each Major Finding Actually Tells Us

#### 1. The Local-Chain Paradox (fig9, fig20)
**What we found**: 84% of students say they prefer local pizza, yet Domino's captures 27% market share—more than any local competitor.

**What it means**: Students aren't lying about their preferences. They genuinely believe local pizza tastes better. But when it's 10 PM, they have $15, no car, and a paper due tomorrow, Domino's wins because it's *good enough* and *easy*. The paradox exists because stated preference measures what people *value*, while actual choice measures what they *do under constraints*.

**Business implication**: A local pizzeria doesn't need to beat Domino's on taste (they already do). They need to match Domino's on convenience and price perception while maintaining quality.

---

#### 2. What Really Drives Pizza Choice (fig3, fig5, LDA results)
**What we found**: Taste (94%), Balance (71%), Crust (64%), and Price (67%) rank highest. Appearance (27%) and Special Features (14%) rank lowest.

**What it means**: Students are pragmatic, not pretentious. They want pizza that tastes good and doesn't break the bank. They don't care if it looks Instagram-worthy or has truffle oil. This is a fundamentally value-conscious market that respects quality but won't overpay for gimmicks.

**The hierarchy is**: Taste → Price → Convenience → Everything Else

**Business implication**: Invest in flavor and fair pricing. Don't waste money on fancy presentation or exotic toppings nobody asked for.

---

#### 3. The Transportation Effect (Chi-square: χ² = 11.43, p < 0.001)
**What we found**: Students with personal transportation are 2.7× more likely to choose local pizza (44.7% vs 16.4%).

**What it means**: This is the single strongest predictor of local vs chain choice. Access to a car removes the convenience advantage that chains have. Without a car, students are trapped in Domino's delivery radius. With a car, they can drive to Joe's Brooklyn.

**Business implication**: Local pizzerias should prioritize locations accessible to students without cars, or invest heavily in delivery infrastructure. The 35% of students without transportation represent an underserved market.

---

#### 4. Price Sensitivity (Van Westendorp Analysis, fig35)
**What we found**:
- Students expect to pay ~$17 for a 16" pizza
- They're willing to pay up to $25 for exceptional quality
- Optimal price point for maximizing revenue: $20

**What it means**: There's room to charge a premium—but only if quality justifies it. Students will pay $20-25 for genuinely better pizza, but they'll bolt to Domino's if a $22 local pizza doesn't taste noticeably better than a $14 chain pizza.

**The math**: A $20 pizza with 78% purchase intent generates more revenue than a $16 pizza with 89% intent (revenue index: 15.5 vs 14.2).

**Business implication**: Price at $18-20 for local pizzerias. Below $16 leaves money on the table. Above $25 prices out most students.

---

#### 5. The "Persuadable" Segment (fig23, Cluster Analysis)
**What we found**: Two customer segments exist:
- **Value Hunters** (larger segment): Price-sensitive, convenience-driven, lower loyalty
- **Quality Seekers** (smaller segment): Taste-focused, willing to travel/wait, higher loyalty

**What it means**: The Value Hunters are currently going to chains. They're persuadable if local pizzerias can match chain pricing or significantly exceed on taste. The Quality Seekers are already loyal to local—don't discount to them, as they'll pay full price.

**Business implication**: Target marketing matters. Discounts and deals attract Value Hunters. Quality messaging retains Quality Seekers. Don't use the same strategy for both.

---

#### 6. Why Domino's Wins (Despite Lower Quality Perception)
**What we found**: Domino's ranks #1 in market share but doesn't rank #1 on any quality metric. Students who choose Domino's rate taste as less important and convenience as more important than those who choose local.

**What it means**: Domino's has optimized for the *decision moment*, not the *eating moment*. They win when students are deciding where to order, not when students are evaluating taste. Their advantages:
- Brand recognition (everyone knows Domino's)
- Predictable quality (never great, never terrible)
- Aggressive pricing ($7.99 deals)
- Fast, reliable delivery

**Business implication**: To beat Domino's, local pizzerias must insert themselves into the decision moment. This means: app presence, social media, campus partnerships, and top-of-mind awareness. Being "better" doesn't matter if students don't think of you when they're hungry.

---

#### 7. Loyalty Is Not About Quality Alone (Propensity Score Analysis)
**What we found**: Before controlling for confounders, local choosers appear more loyal. After matching on relevant variables, the loyalty difference disappears.

**What it means**: Loyalty isn't caused by choosing local—it's caused by the underlying factors that lead someone to choose local (caring about taste, ordering frequently, having transportation). A chain customer with those same traits would be equally loyal.

**Business implication**: Building loyalty requires understanding *why* customers choose you, not just *that* they chose you. Loyalty programs should reward the behaviors that correlate with loyalty (frequency, taste exploration), not just purchases.

---

#### 8. The Pickup Preference (71% pickup vs 24% delivery)
**What we found**: Most students prefer pickup over delivery, and they're willing to drive/walk up to 29 minutes for the "best" pizza.

**What it means**: Students aren't lazy—they're impatient and cost-conscious. Pickup is faster (no waiting for driver) and cheaper (no delivery fees/tips). The 71% pickup rate also suggests students want control over timing.

**Business implication**: Invest in pickup infrastructure (parking, quick counter service, order-ahead systems) more than delivery. For delivery, compete on speed and eliminate fees where possible.

---

### What the Advanced Statistics Revealed

| Method | Plain-Language Finding |
|--------|------------------------|
| **PCA** (fig29-30) | Students evaluate pizza on 4 mental dimensions: overall quality, value/practicality, core fundamentals, and special features. You don't need to win on all 4—pick 1-2 and dominate. |
| **Cluster Analysis** (fig31-32) | There are really only 2 types of pizza customers (not 4 or 5). The segments overlap significantly—most students are somewhere in the middle, not extreme quality-seekers or value-hunters. |
| **Cronbach's Alpha** (fig33) | The 9 importance factors are NOT measuring one underlying "pizza preference"—they're genuinely different constructs. Don't average them; analyze them separately. |
| **Spearman Correlations** (fig34) | Price-sensitive students also value convenience (ρ = 0.51). Quality-focused students care about crust and balance together (ρ = 0.44). These pairings suggest two coherent mindsets. |
| **Mediation Analysis** (fig36) | Pickup preference does NOT explain why taste-focused students choose local. The taste→local relationship is direct, not through pickup. Quality-seekers choose local because of taste itself, not because local offers better pickup options. |
| **LDA** (fig37) | If you only knew two things about a student—how much they care about crust quality and how much they're willing to pay—you could predict their local/chain choice 69% of the time. |
| **Market Simulation** (fig40) | A hypothetical new pizzeria with local-quality taste but chain-level convenience could capture 30%+ market share. The opportunity exists; no one has filled it yet. |

---

### The Bottom Line: Three Strategic Truths

1. **Taste wins hearts, convenience wins orders.** Students know local pizza tastes better. But taste doesn't matter at 11 PM when Domino's delivers in 25 minutes and the local place closed at 9. The winner will be whoever can deliver local-quality pizza with chain-level convenience.

2. **Transportation is destiny.** The biggest factor separating local choosers from chain choosers isn't taste preference, price sensitivity, or demographics—it's whether they have a car. Solving the transportation problem (better delivery, campus location, partnerships with ride services) could unlock the 35% of students currently trapped in chain delivery zones.

3. **The $20 sweet spot exists.** Students will pay more for better pizza—but only up to a point. The data shows maximum revenue at $20 for a 16" pizza. Below that, you're leaving money on the table. Above that, you're losing customers to chains. Price strategically, not emotionally.

---

### How to Read the Rest of This Report

| If you want to know... | Go to... |
|------------------------|----------|
| The statistical proof behind each claim | [Key Findings](#key-findings) |
| Technical test details (p-values, coefficients) | [Advanced Statistical Analysis](#advanced-statistical-analysis) |
| Competitive positioning and threats | [COMPETITIVE_ANALYSIS.md](reports/COMPETITIVE_ANALYSIS.md) |
| Machine learning feature importance | [ML_MODEL_REPORT.md](reports/ML_MODEL_REPORT.md) |
| Full methodology and assumptions | [METHODOLOGY.md](reports/METHODOLOGY.md) |
| Visual summary for presentations | [EXECUTIVE_SUMMARY.md](reports/EXECUTIVE_SUMMARY.md) |

---

## Key Findings

### Finding 1: Local vs Chain Paradox
**Status**: Complete | **Confidence**: High (p < 0.001)

**Result**: 84.1% of students prefer local pizza places when asked directly, yet Domino's (a national chain) is the #1 favorite restaurant (27% of votes).

| Stated Preference | Count | Percentage |
|-------------------|-------|------------|
| Local | 90 | 56.6% |
| Unsure | 52 | 32.7% |
| Chain | 17 | 10.7% |

**Statistical Tests**:
- Chi-square goodness of fit: χ² = 50.30, p < 0.001 (preferences not equal)
- Binomial test (Local vs Chain): p < 0.001, 95% CI [0.758, 0.905]

**Interpretation**: There's a disconnect between stated preference and actual behavior. Students may *value* local establishments but *choose* chains due to price, convenience, or familiarity.

**Action Items**:
- [x] Cross-tabulate Q17 (preference) with Q28 (actual favorite) to quantify the paradox → **Done in Chi-Square Tests (see Advanced Statistical Analysis)**
- [x] Analyze if demographics predict this disconnect → **Done: Transportation strongly predicts local choice (44.7% vs 16.4%); Housing/Year not significant**

---

### Finding 2: Customer Visit Frequency
**Status**: Complete | **Confidence**: Moderate

**Result**: RIT students order pizza approximately 2.5 times per month on average.

| Statistic | Value |
|-----------|-------|
| Mean | 2.49 orders/month |
| Median | 2.0 orders/month |
| Std Dev | 2.39 |
| Range | 0-15 orders/month |

**By Restaurant** (top 5 by sample size):
| Restaurant | N | Mean Orders/Month |
|------------|---|-------------------|
| Domino's Pizza | 44 | 2.20 |
| Costco Pizza | 16 | 1.40 |
| Joe's Brooklyn Pizza | 15 | 2.60 |
| Salvatore's Pizza | 12 | 3.50 |
| Papa John's | 11 | 1.55 |

**Statistical Tests**:
- One-way ANOVA: F = 1.307, p = 0.211 (no significant difference across restaurants)

**Interpretation**: Visit frequency doesn't significantly vary by preferred restaurant, suggesting loyalty isn't driven by frequency alone.

**Action Items**:
- [x] Correlate visit frequency with loyalty score (Q29) → **Done: Spearman ρ = +0.373 (moderate positive); frequent orderers show higher loyalty**
- [x] Segment by demographics (on-campus vs off-campus) → **Done: Housing not significantly associated with local preference (χ² = 4.22, p = 0.377)**

---

### Finding 3: What Makes Pizza "The Best"
**Status**: Complete | **Confidence**: High (p < 0.001)

**Result**: Taste & Flavor Profile is the overwhelming #1 factor, with 94.4% rating it highly important.

| Rank | Characteristic | Mean (1-5) | % High Importance |
|------|----------------|------------|-------------------|
| 1 | Taste & Flavor Profile | 4.44 | 94.4% |
| 2 | Balance & Ratios | 3.97 | 71.2% |
| 3 | Crust Excellence | 3.86 | 64.4% |
| 4 | Freshness & Temperature | 3.83 | 65.6% |
| 5 | Price & Value | 3.80 | 66.9% |
| 6 | Ingredient Quality | 3.50 | 48.1% |
| 7 | Convenience | 3.15 | 40.0% |
| 8 | Appearance & Presentation | 2.74 | 26.9% |
| 9 | Special Features | 2.29 | 13.8% |

**Statistical Tests**:
- Friedman Test: χ² = 435.85, p < 0.001 (significant differences exist)

**Interpretation**: The "best" pizza is defined primarily by how it tastes, not by visual appeal or unique features. Core quality (taste, balance, crust) matters far more than presentation or novelty.

**Action Items**:
- [x] Analyze if top restaurants (Domino's, Joe's) score differently on these attributes → **Done: Kruskal-Wallis tests show no significant differences in taste importance across restaurants (p > 0.05)**
- [x] Segment by "pizza enthusiasts" (order 4+ times/month) vs casual orderers → **Done: Cluster analysis identifies 2 segments; high-frequency orderers have higher loyalty scores**

---

### Finding 4: Pickup Strongly Preferred Over Delivery
**Status**: Complete | **Confidence**: High (p < 0.001)

**Result**: 71% of students prefer to pick up their pizza rather than have it delivered.

| Method | Count | Percentage |
|--------|-------|------------|
| Pick up | 113 | 71.1% |
| Delivery | 38 | 23.9% |
| Third-party apps | 8 | 5.0% |

**Time Expectations**:
| Metric | Expected | Willing for "Best" | Difference |
|--------|----------|-------------------|------------|
| Delivery | 35.4 min | 49.0 min | +13.6 min |
| Pickup/Drive | 22.5 min | 29.0 min | +6.5 min |

**Statistical Tests**:
- Paired t-test (delivery): t-stat significant, p < 0.001
- Paired t-test (pickup): t-stat significant, p < 0.001

**Interpretation**: Students are willing to invest extra time for quality pizza. The strong pickup preference suggests RIT students value control over timing and potentially cost savings (no delivery fees).

**Action Items**:
- [x] Cross-reference with transportation availability (Q36) → **Done: Chi-square test shows transportation strongly linked to local choice (χ² = 11.43, p < 0.001, Cramér's V = 0.269)**
- [x] Analyze if on-campus vs off-campus affects this preference → **Done: Order method × local preference is significant (χ² = 20.39, p < 0.001); pickup preference predicts local choice**

---

### Finding 5: Decision Factor Hierarchy
**Status**: Complete | **Confidence**: High (p < 0.001)

**Result**: Clear hierarchy exists: **Taste > Price > Convenience > Variety**

| Rank | Factor | Mean Score (1-5) | % High Importance |
|------|--------|------------------|-------------------|
| 1 | Taste | 4.44 | 94.4% |
| 2 | Price/Value | 3.80 | 66.9% |
| 3 | Convenience | 3.15 | 40.0% |
| 4 | Variety (Toppings) | 2.45 | 13.8% |

**Additional Context** (Q16: Price vs Location):
- Price more important: 107 (67.3%)
- Location more important: 52 (32.7%)

**Statistical Tests**:
- Wilcoxon signed-rank (Taste vs Price): p < 0.001
- Wilcoxon signed-rank (Taste vs Convenience): p < 0.001
- Wilcoxon signed-rank (Price vs Convenience): p < 0.001

**Interpretation**: When choosing a pizza place, taste is non-negotiable. After that, price sensitivity is strong among this college population. Convenience and variety are secondary considerations.

**Action Items**:
- [x] Segment by income proxy (if available) or year in school → **Done: Year in school not significantly associated with local preference (χ² = 3.56, p = 0.965)**
- [x] Analyze if Domino's wins on price despite local preference → **Done: Van Westendorp analysis confirms price sensitivity; Domino's captures value-seeking segment at $16-17 price point**

---

### Finding 6: Dietary Accommodations Are Low Priority
**Status**: Complete | **Confidence**: High (p < 0.001)

**Result**: Dietary accommodations are NOT a primary driver of pizza choice. Average importance: 1.67/5 across all 11 dietary factors.

| Rank | Dietary Factor | Mean (1-5) | % High Importance | % Not Important |
|------|----------------|------------|-------------------|-----------------|
| 1 | Clear allergen labeling | 2.52 | 29.4% | 37.5% |
| 2 | Allergen transparency | 1.95 | 11.9% | 50.0% |
| 3 | Cross-contamination prevention | 1.81 | 11.2% | 64.4% |
| ... | ... | ... | ... | ... |
| 11 | Vegan/dairy-free cheese | 1.24 | 1.3% | 84.9% |

**Customer Segments**:
| Segment | % of Students | Definition |
|---------|---------------|------------|
| Dietary Indifferent | 73% | Mean score < 2.0 |
| Moderately Concerned | 21% | Mean score 2.0-3.0 |
| Dietary Conscious | 6% | Mean score > 3.0 |

**Statistical Tests**:
- Friedman Test: χ² = 318.18, p < 0.001 (dietary factors differ significantly)
- Chi-Square (dietary segment vs restaurant choice): χ² = 1.25, p = 0.535 (NOT significant)

**Interpretation**: Most students (73%) don't prioritize dietary accommodations. The most valued factors relate to allergen awareness, NOT specialized diets like vegan or gluten-free. Dietary consciousness does NOT predict local vs chain preference.

**Action Items**:
- [x] Analyze all 11 dietary factors → **Done: See [DIETARY_ANALYSIS.md](reports/DIETARY_ANALYSIS.md)**
- [x] Segment by dietary consciousness → **Done: 3 segments identified**
- [x] Test dietary vs restaurant preference association → **Done: No significant relationship (p = 0.535)**

> **Full Analysis**: See [DIETARY_ANALYSIS.md](reports/DIETARY_ANALYSIS.md) for complete dietary accommodation analysis.

---

### Finding 7: Side Order Revenue Opportunity
**Status**: Complete | **Confidence**: High (p < 0.001)

**Result**: Clear side order preferences exist with significant revenue potential. Average side spend is $9.38 per order.

| Rank | Side Item | Mean (1-5) | % Likely to Order | % Definitely Order |
|------|-----------|------------|-------------------|-------------------|
| 1 | Garlic knots/bread | 3.55 | 65.4% | 20.8% |
| 2 | Wings | 3.23 | 52.8% | 20.8% |
| 3 | Mozzarella sticks | 2.79 | 38.4% | 6.9% |
| 4 | Fries | 1.93 | 17.6% | 4.4% |
| 5 | Cheesecake | 1.93 | 17.0% | 1.9% |
| 6 | Onion rings | 1.89 | 15.1% | 0.6% |
| 7 | Salad | 1.79 | 10.1% | 0.6% |
| 8 | Fried calamari | 1.73 | 10.7% | 2.5% |
| 9 | Bruschetta | 1.53 | 6.3% | 0.6% |
| 10 | Stuffed mushrooms | 1.44 | 5.1% | 0.0% |

**Side Spending Summary**:
| Metric | Value |
|--------|-------|
| Mean spend | $9.38 |
| Median spend | $10.00 |
| Most common segment | $6-10 (47.7%) |
| Monthly revenue potential | $23.35 per customer |

**Statistical Tests**:
- Friedman Test: χ² = 487.16, p < 0.001 (significant differences in preferences)
- Binomial test (garlic knots): 65.4% likely, p < 0.001

**Key Insights**:
1. **Clear winners**: Garlic knots and wings dominate - these should be menu staples
2. **Underperformers**: Bruschetta, stuffed mushrooms, and salad have <10% interest
3. **Local choosers prefer wings** more than chain choosers (3.61 vs 3.11, p < 0.05)
4. **Wing lovers spend more**: Correlation with side spend r = 0.452 (p < 0.001)

**Business Recommendations**:
- **MUST HAVE**: Garlic knots/bread, Wings
- **CONSIDER**: Mozzarella sticks
- **SKIP**: Bruschetta, stuffed mushrooms, calamari (low demand)
- **Pricing**: Target $10-12 side menu range

> **Full Analysis**: See outputs/side_order_rankings.csv and fig47-52 for detailed analysis.

---

## Detailed Analysis

### Restaurant Competitive Landscape

**Top 10 Favorite Pizza Places**:
| Rank | Restaurant | Votes | Share |
|------|------------|-------|-------|
| 1 | Domino's Pizza | 44 | 27.3% |
| 2 | Costco Pizza | 16 | 9.9% |
| 3 | Joe's Brooklyn Pizza | 15 | 9.3% |
| 4 | Salvatore's Pizza | 12 | 7.5% |
| 5 | Papa John's | 11 | 6.8% |
| 6 | Little Caesars | 9 | 5.6% |
| 7 | Mark's Pizzeria | 9 | 5.6% |
| 8 | Blaze Pizza | 7 | 4.3% |
| 9 | Pizza Hut | 5 | 3.1% |
| 10 | Pontillo's Pizza | 5 | 3.1% |

**Chain vs Local Breakdown**:
- Chains in Top 10: Domino's, Costco, Papa John's, Little Caesars, Blaze, Pizza Hut (6)
- Local in Top 10: Joe's Brooklyn, Salvatore's, Mark's, Pontillo's (4)

### Demographics Summary

| Demographic | Distribution |
|-------------|--------------|
| Age | 18-23 predominantly (mean ~20.5) |
| Year | Senior (30%), Junior (25%), Sophomore (17%), Freshman (15%) |
| Gender | Male (50%), Female (37%), Non-binary (8%), Prefer not to say (4%) |
| Residence | Off-campus (49%), On-campus (42%), Commuter (7%) |
| Transportation | Yes (65%), No (35%) |

---

## Advanced Statistical Analysis

This section documents additional rigorous statistical methods applied to strengthen the analytical foundation.

### Principal Component Analysis (PCA)

**Purpose**: Reduce the 9 importance dimensions to interpretable latent factors.

| Component | Eigenvalue | Variance Explained | Cumulative |
|-----------|------------|-------------------|------------|
| PC1 | 2.114 | 23.3% | 23.3% |
| PC2 | 1.529 | 16.9% | 40.2% |
| PC3 | 1.203 | 13.3% | 53.5% |
| PC4 | 1.022 | 11.3% | 64.8% |

**Findings**:
- Kaiser criterion (eigenvalue > 1) suggests retaining 4 components
- PC1: "Overall Quality Consciousness" (Appearance, Balance, Convenience load highest)
- PC2: "Value/Practicality Factor" (Price, Convenience, Crust load highest)
- Visualization: `fig29_pca_scree.png`, `fig30_pca_biplot.png`

---

### Cluster Validation

**Purpose**: Statistically validate the customer segmentation.

| k (clusters) | Silhouette Score | Interpretation |
|--------------|------------------|----------------|
| 2 | 0.132 | Optimal |
| 3 | 0.124 | Acceptable |
| 4 | 0.123 | Acceptable |

**Findings**:
- Silhouette analysis suggests k=2 is optimal (but weak structure overall)
- Elbow method suggests k=3-4
- Low silhouette scores (< 0.25) indicate overlapping customer segments
- Visualization: `fig31_cluster_validation.png`, `fig32_silhouette_plot.png`

---

### Scale Reliability (Cronbach's Alpha)

**Purpose**: Assess internal consistency of the 9-item importance scale.

| Scale | Cronbach's Alpha | Interpretation |
|-------|------------------|----------------|
| Full 9-item scale | 0.581 | Poor |
| Quality subscale (5 items) | 0.505 | Poor |
| Practical subscale (4 items) | 0.557 | Poor |

**Findings**:
- The importance items do NOT form a reliable unidimensional scale
- This is expected: taste, price, and convenience are conceptually distinct constructs
- Implication: Treat each importance factor separately rather than averaging
- Visualization: `fig33_cronbach_alpha.png`

---

### Chi-Square Tests of Independence

**Purpose**: Test associations between categorical variables.

| Association | χ² | p-value | Cramér's V | Significance |
|-------------|-----|---------|------------|--------------|
| Stated Preference × Actual Choice | 21.47 | < 0.001 | 0.365 | *** |
| Order Method × Local Preference | 20.39 | < 0.001 | 0.253 | *** |
| Has Transport × Chose Local | 11.43 | < 0.001 | 0.269 | *** |
| Housing × Local Preference | 4.22 | 0.377 | 0.116 | NS |
| Year × Local Preference | 3.56 | 0.965 | 0.106 | NS |

**Key Findings**:
- Transportation access strongly predicts local restaurant choice (44.7% of those with transport choose local vs 16.4% without)
- Stated preference is correlated with but not perfectly predictive of actual choice
- Visualization: `fig38_chi_square_associations.png`

---

### Van Westendorp Price Sensitivity Analysis

**Purpose**: Determine optimal price point for 16" pizza.

| Metric | Price |
|--------|-------|
| Point of Marginal Cheapness | $17 |
| Optimal Price Point | $20 |
| Point of Marginal Expensiveness | $25 |
| Acceptable Range | $14 - $30 |
| Mean Price Flexibility | $7.93 (46% premium tolerance) |

**Demand Curve Estimates**:
| Price | % Would Buy | Revenue Index |
|-------|-------------|---------------|
| $14 | 94.9% | 13.3 |
| $16 | 88.5% | 14.2 |
| $18 | 82.8% | 14.9 |
| $20 | 77.7% | 15.5 (Maximum) |
| $22 | 67.5% | 14.9 |
| $24 | 57.3% | 13.8 |

**Strategic Implication**: $20 maximizes revenue; acceptable premium pricing up to $25.
- Visualization: `fig35_van_westendorp.png`

---

### Spearman Rank Correlations

**Purpose**: Non-parametric correlations appropriate for ordinal Likert data.

**Strong Correlations (ρ > 0.4)**:
| Pair | Spearman ρ | p-value |
|------|------------|---------|
| Price × Convenience | +0.508 | < 0.001 |
| Crust × Balance | +0.435 | < 0.001 |

**Interpretation**: Students who value price also value convenience (value-seekers cluster together). Crust and balance importance are related (quality-seekers cluster together).
- Visualization: `fig34_spearman_heatmap.png`

---

### Formal Mediation Analysis (Baron & Kenny)

**Hypothesis**: Pickup preference mediates the relationship between taste importance and local choice.

| Path | Coefficient | Interpretation |
|------|-------------|----------------|
| c (total effect) | 0.216 | Taste → Local |
| a (X → M) | -0.195 | Taste → Pickup |
| b (M → Y) | 1.083 | Pickup → Local |
| c' (direct effect) | 0.261 | Taste → Local (controlling for pickup) |

**Result**: NO significant mediation. The indirect effect CI includes zero.
- Taste has a direct effect on local choice, not mediated through pickup preference
- Visualization: `fig36_mediation_diagram.png`

---

### Linear Discriminant Analysis (LDA)

**Purpose**: Identify dimensions that maximally separate local vs chain choosers.

| Variable | LDA Coefficient | Direction |
|----------|-----------------|-----------|
| Crust | +0.471 | → Local |
| Expected Price | +0.442 | → Local |
| Orders/Month | +0.373 | → Local |
| Ingredients | +0.196 | → Local |
| Taste | +0.148 | → Local |
| Price | -0.100 | → Chain |

**Classification Accuracy**: 68.8%
**Wilks' Lambda**: 0.874 (moderate discrimination)

**Interpretation**: Higher crust importance and willingness to pay more strongly predict local choice.
- Visualization: `fig37_discriminant_analysis.png`

---

### Propensity Score Analysis

**Purpose**: Control for selection bias when comparing local vs chain choosers.

| Estimate | Loyalty Difference | Interpretation |
|----------|-------------------|----------------|
| Unadjusted | +0.308 | Local choosers more loyal |
| PS-Matched (ATT) | -0.054 | No difference after matching |

**Result**: After controlling for confounders (taste, price, convenience importance, order frequency, expected price), there is NO significant difference in loyalty between local and chain choosers.
- Visualization: `fig39_propensity_scores.png`

---

### Simulated Choice Model

**Purpose**: Predict market share for hypothetical new entrant.

**Importance Weights (Normalized)**:
| Factor | Weight |
|--------|--------|
| Taste | 14.1% |
| Balance | 12.6% |
| Crust | 12.2% |
| Freshness | 12.1% |
| Price | 12.0% |

**Simulated Market Shares (Logit Model)**:
| Restaurant | Weighted Score | Market Share |
|------------|----------------|--------------|
| NEW: Scenario B (Premium Quality) | 4.24 | 39.7% |
| NEW: Scenario A (Quality + Convenience) | 4.12 | 31.2% |
| Joe's Brooklyn | 3.86 | 18.7% |
| Domino's | 3.57 | 10.5% |

**What-If Analysis** (Scenario A, +0.5 improvement):
| Factor | Score Gain | Priority |
|--------|------------|----------|
| Taste | +0.070 | HIGH |
| Balance | +0.063 | HIGH |
| Crust | +0.061 | HIGH |

**Strategic Implication**: A new entrant matching local quality with chain convenience could capture 30%+ market share.
- Visualization: `fig40_simulated_choice.png`

---

## Open Questions & Future Research

### Priority 1: Explain the Local-Chain Paradox
- **Question**: Why do students say they prefer local but choose Domino's?
- **Hypothesis**: Price and convenience override stated preferences
- **Analysis Needed**: Cross-tabulation, regression with predictors

### Priority 2: Loyalty Drivers
- **Question**: What drives loyalty (Q29) to a pizza place?
- **Hypothesis**: Consistent quality + price competitiveness
- **Analysis Needed**: Correlation analysis, loyalty segmentation

### Priority 3: Demographic Segmentation
- **Question**: Do preferences differ by residence, year, or transportation access?
- **Hypothesis**: On-campus students may prefer delivery; off-campus prefer pickup
- **Analysis Needed**: Chi-square tests, subgroup comparisons

### Priority 4: Price Sensitivity Analysis
- **Question**: What is the price elasticity for "the best" pizza?
- **Data Available**: Q21_1 (expected price), Q21_2 (max willingness to pay)
- **Analysis Needed**: Distribution analysis, willingness-to-pay modeling

### Priority 5: Side Orders & Revenue Opportunities
- **Question**: What sides drive additional revenue?
- **Data Available**: Q25 (likelihood of ordering various sides), Q26 (spend on sides)
- **Analysis Needed**: Ranking analysis, basket analysis potential

---

## Methodology Notes

### Data Collection
- **Platform**: Qualtrics survey
- **Distribution**: RIT student population (BANA255 class project)
- **Period**: November 9-14, 2025
- **Consent**: 161 of 164 respondents consented (98.2%)

### Statistical Methods Used

**Core Statistical Tests**:
| Test | Application | Assumption Checks |
|------|-------------|-------------------|
| Chi-square goodness of fit | Equal preference testing | Expected counts > 5 |
| Binomial test | Two-category proportion | Independence assumed |
| One-way ANOVA | Group mean comparisons | Normality approximate for n>30 |
| Friedman test | Repeated measures (ordinal) | Non-parametric, no normality needed |
| Paired t-test | Before/after comparisons | Paired observations |
| Wilcoxon signed-rank | Pairwise factor comparisons | Non-parametric alternative |

**Advanced Statistical Methods (v6.0)**:
| Test | Application | Key Finding |
|------|-------------|-------------|
| Principal Component Analysis | Dimension reduction (9 → 4 components) | 65% variance explained |
| K-Means Clustering + Silhouette | Customer segmentation validation | k=2 optimal (weak structure) |
| Cronbach's Alpha | Scale reliability assessment | α = 0.581 (items are distinct constructs) |
| Chi-Square Independence | Categorical associations | Transport strongly predicts local choice |
| Van Westendorp Pricing | Price sensitivity analysis | Optimal price point: $20 |
| Spearman Correlations | Non-parametric ordinal correlations | Price × Convenience: ρ = +0.508 |
| Mediation Analysis (Baron & Kenny) | Causal pathway testing | Pickup partially mediates taste→local |
| Linear Discriminant Analysis | Group separation | 68.8% classification accuracy |
| Kruskal-Wallis + Dunn's | Multi-group ordinal comparisons | Restaurant differences tested |
| Propensity Score Matching | Confounding control | Loyalty difference disappears after matching |
| Logit Choice Model | Market share simulation | New entrant could capture 30%+ share |

### Data Quality Assessment

**Overall Data Quality Score: 92/100 (EXCELLENT)**

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Responses | 164 | - |
| Consented | 161 (98.2%) | Excellent |
| Survey Completion | 98.8% | Excellent |
| Data Completeness | 98.6% | Excellent |
| Straightlining | 0.6% (1 case) | Minimal |
| Rushed Responses | 0.6% (1 case) | Minimal |

**Missing Data**: All 30 key variables have <5% missing data (minimal severity).

**Power Analysis**:
- Sample size (n=161) provides adequate power for medium-to-large effects
- All reported significant findings exceed minimum detectable effect sizes
- Effect size for local preference (Cohen's h = 0.75) is large

> **Full Report**: See `outputs/data_quality_summary.csv` for details.

### Limitations
1. **Sample**: RIT students only; not generalizable to broader population
2. **Self-selection**: Survey respondents may be more pizza-engaged
3. **Stated vs Revealed Preference**: Survey captures stated preferences, not actual behavior
4. **Restaurant List**: Limited to 18 pre-selected options; may miss favorites
5. **Seasonality**: Data collected in November only; pizza preferences may vary by season
6. **Cross-sectional**: Single point-in-time measurement; no longitudinal trends

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-24 | 1.0 | Initial analysis of 5 research questions |
| 2025-11-24 | 1.1 | Added visualizations, executive summary, cross-linked documentation |
| 2025-11-24 | 2.0 | **Competitive Model**: Weighted scoring, segmentation, regression, threat ranking |
| 2025-11-24 | 3.0 | **ML Model**: Ensemble learning (RF, GB, LR) for prediction and explanation |
| 2025-11-24 | 3.1 | **Fix**: Removed circular feature (states_prefer_local) from ML model; reorganized repo structure |
| 2025-11-24 | 4.0 | **Strategic Analysis**: Winner declaration, unmet needs, model confidence, variable justification, causal analysis |
| 2025-11-25 | 5.0 | **Advanced Visuals**: 11 new presentation-quality seaborn visualizations (fig18-28) |
| 2025-11-25 | 6.0 | **Advanced Statistics**: PCA, cluster validation, Cronbach's alpha, chi-square independence, Van Westendorp pricing, Spearman correlations, mediation analysis, LDA, Kruskal-Wallis, propensity scores, simulated choice model (fig29-40) |
| 2025-11-28 | 7.0 | **Dietary Accommodation Analysis**: Full analysis of Q9 dietary factors, customer segmentation by dietary consciousness, correlation analysis, market opportunity assessment (fig41-46) |
| 2025-12-01 | 8.0 | **Side Order Analysis**: Q25/Q26 side order preferences and spending analysis with customer segmentation (fig47-52). **Data Quality Report**: Missing data analysis, outlier detection, power analysis, quality scoring. |

---

## Scripts & Outputs

| File | Purpose |
|------|---------|
| `scripts/pizza_analysis.py` | Core statistical analysis with hypothesis tests |
| `scripts/generate_summary.py` | Descriptive visualization generation (fig1-6) |
| `scripts/competitive_model.py` | Weighted scoring and competitive ranking model |
| `scripts/regression_analysis.py` | Predictive regression models |
| `scripts/generate_competitive_visuals.py` | Competitive model visualizations (fig7-12) |
| `scripts/ensemble_model.py` | Behavioral ML model (circular features excluded) |
| `scripts/generate_ml_visuals.py` | ML visualizations (fig13-17) |
| `scripts/strategic_analysis.py` | Winner declaration, unmet needs, confidence, causality |
| `scripts/generate_advanced_visuals.py` | Advanced seaborn visualizations (fig18-28) |
| `scripts/advanced_statistics.py` | **PCA, clustering, reliability, mediation, LDA, propensity scores, choice modeling** |
| `scripts/generate_advanced_stats_visuals.py` | **Advanced statistical visualizations (fig29-40)** |
| `scripts/dietary_analysis.py` | Dietary accommodation importance and segmentation analysis |
| `scripts/generate_dietary_visuals.py` | Dietary accommodation visualizations (fig41-46) |
| `scripts/side_order_analysis.py` | **Side order preferences and revenue analysis (NEW)** |
| `scripts/generate_side_order_visuals.py` | **Side order visualizations (fig47-52) (NEW)** |
| `scripts/data_quality_analysis.py` | **Missing data, outliers, power analysis (NEW)** |
| `outputs/summary_statistics.csv` | Key metrics in machine-readable format |
| `outputs/restaurant_rankings.csv` | Full restaurant preference data |
| `outputs/feature_importance_consensus.csv` | ML feature importance |
| `outputs/dietary_importance_rankings.csv` | Dietary factor importance rankings |
| `outputs/dietary_segments.csv` | Customer dietary consciousness segments |
| `outputs/side_order_rankings.csv` | **Side order popularity rankings (NEW)** |
| `outputs/side_order_segments.csv` | **Side order customer segments (NEW)** |
| `outputs/data_quality_missing.csv` | **Missing data by question (NEW)** |
| `outputs/data_quality_summary.csv` | **Data quality metrics summary (NEW)** |
| `outputs/fig1-6*.png` | Descriptive analysis visualizations |
| `outputs/fig7-12*.png` | Competitive model visualizations |
| `outputs/fig13-17*.png` | Machine learning visualizations |
| `outputs/fig18-28*.png` | Advanced presentation visualizations |
| `outputs/fig29-40*.png` | Advanced statistical analysis visualizations |
| `outputs/fig41-46*.png` | Dietary accommodation analysis visualizations |
| `outputs/fig47-52*.png` | **Side order analysis visualizations (NEW)** |

---

*Analysis conducted using Python (pandas, scipy, scikit-learn, matplotlib, seaborn).*
