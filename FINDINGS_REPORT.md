# BANA255 Pizza Survey - Findings Report

**Study**: What is the Best Pizza Available to RIT Students?
**Data Collected**: November 9-14, 2025
**Valid Responses**: 161 (consented participants)
**Last Updated**: November 24, 2025

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | Visual summary with charts and key takeaways |
| [METHODOLOGY.md](METHODOLOGY.md) | Statistical methods, assumptions, and references |
| `outputs/` | Generated figures and CSV tables |

### Generated Visualizations
- `fig1_local_vs_chain.png` - Preference distribution
- `fig2_top_pizza_places.png` - Restaurant rankings
- `fig3_importance_ratings.png` - Quality characteristic importance
- `fig4_order_time_preferences.png` - Order method & time analysis
- `fig5_decision_factors.png` - Key decision factors
- `fig6_demographics.png` - Sample demographics

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Key Findings](#key-findings)
3. [Detailed Analysis](#detailed-analysis)
4. [Open Questions & Future Research](#open-questions--future-research)
5. [Methodology Notes](#methodology-notes)

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
- [ ] Cross-tabulate Q17 (preference) with Q28 (actual favorite) to quantify the paradox
- [ ] Analyze if demographics predict this disconnect

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
- [ ] Correlate visit frequency with loyalty score (Q29)
- [ ] Segment by demographics (on-campus vs off-campus)

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
- [ ] Analyze if top restaurants (Domino's, Joe's) score differently on these attributes
- [ ] Segment by "pizza enthusiasts" (order 4+ times/month) vs casual orderers

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
- [ ] Cross-reference with transportation availability (Q36)
- [ ] Analyze if on-campus vs off-campus affects this preference

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
- [ ] Segment by income proxy (if available) or year in school
- [ ] Analyze if Domino's wins on price despite local preference

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
| Test | Application | Assumption Checks |
|------|-------------|-------------------|
| Chi-square goodness of fit | Equal preference testing | Expected counts > 5 |
| Binomial test | Two-category proportion | Independence assumed |
| One-way ANOVA | Group mean comparisons | Normality approximate for n>30 |
| Friedman test | Repeated measures (ordinal) | Non-parametric, no normality needed |
| Paired t-test | Before/after comparisons | Paired observations |
| Wilcoxon signed-rank | Pairwise factor comparisons | Non-parametric alternative |

### Limitations
1. **Sample**: RIT students only; not generalizable to broader population
2. **Self-selection**: Survey respondents may be more pizza-engaged
3. **Stated vs Revealed Preference**: Survey captures stated preferences, not actual behavior
4. **Restaurant List**: Limited to 18 pre-selected options; may miss favorites

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-24 | 1.0 | Initial analysis of 5 research questions |
| 2025-11-24 | 1.1 | Added visualizations, executive summary, cross-linked documentation |

---

## Scripts & Outputs

| File | Purpose |
|------|---------|
| `pizza_analysis.py` | Core statistical analysis with hypothesis tests |
| `generate_summary.py` | Visualization and summary table generation |
| `outputs/summary_statistics.csv` | Key metrics in machine-readable format |
| `outputs/restaurant_rankings.csv` | Full restaurant preference data |
| `outputs/fig*.png` | Publication-ready visualizations |

---

*Analysis conducted using Python (pandas, scipy, matplotlib, seaborn).*
