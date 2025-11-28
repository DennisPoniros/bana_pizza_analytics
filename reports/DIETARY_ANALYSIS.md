# Dietary Accommodation Analysis Report

**Purpose**: Analyze dietary accommodation preferences and their impact on pizza choice among RIT students.

**Date**: November 2025 | **Script**: `scripts/dietary_analysis.py`, `scripts/generate_dietary_visuals.py`

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Dietary Importance Rankings](#2-dietary-importance-rankings)
3. [Customer Segmentation](#3-customer-segmentation)
4. [Correlation Analysis](#4-correlation-analysis)
5. [Market Opportunity](#5-market-opportunity)
6. [Demographic Insights](#6-demographic-insights)
7. [Strategic Recommendations](#7-strategic-recommendations)

---

## 1. Executive Summary

### Key Finding

**Dietary accommodations are NOT a primary driver of pizza choice for the majority of RIT students.**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Overall Dietary Importance | **1.67/5** | Low priority |
| % Dietary Indifferent | **73%** | Majority don't prioritize |
| % Moderately Concerned | **21%** | Secondary concern |
| % Dietary Conscious | **6%** | Niche segment |

### Top 3 Dietary Factors (by importance)

| Rank | Factor | Mean Score | % High Importance |
|------|--------|------------|-------------------|
| 1 | Clear allergen labeling | 2.52/5 | 29.4% |
| 2 | Allergen transparency | 1.95/5 | 11.9% |
| 3 | Cross-contamination prevention | 1.81/5 | 11.2% |

### Bottom Line

> **Most students (73%) don't care about dietary accommodations when ordering pizza.** However, a meaningful minority (~27%) does, and the most valued accommodations relate to **allergen awareness**, not specialized diets like vegan or gluten-free.

---

## 2. Dietary Importance Rankings

### Full Rankings (11 Factors)

| Rank | Dietary Consideration | Mean | Median | % High | % Not Important |
|------|----------------------|------|--------|--------|-----------------|
| 1 | Clear allergen labeling | 2.52 | 2.0 | 29.4% | 37.5% |
| 2 | Allergen transparency | 1.95 | 1.5 | 11.9% | 50.0% |
| 3 | Cross-contamination prevention | 1.81 | 1.0 | 11.2% | 64.4% |
| 4 | Organic/non-GMO ingredients | 1.79 | 1.0 | 10.0% | 57.5% |
| 5 | Reduced sodium/healthier options | 1.75 | 1.0 | 7.5% | 55.6% |
| 6 | Half-and-half pizza option | 1.74 | 1.0 | 13.1% | 66.9% |
| 7 | Halal/kosher certification | 1.51 | 1.0 | 8.1% | 75.6% |
| 8 | Plant-based protein toppings | 1.39 | 1.0 | 4.4% | 76.1% |
| 9 | Gluten-free crust options | 1.31 | 1.0 | 1.9% | 81.9% |
| 10 | Low-carb/keto options | 1.31 | 1.0 | 1.9% | 80.0% |
| 11 | Vegan/dairy-free cheese | 1.24 | 1.0 | 1.3% | 84.9% |

**Visualization**: `fig41_dietary_importance.png`, `fig42_dietary_distribution.png`

### Statistical Validation

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Friedman Test | χ² = 318.18 | < 0.001 | Significant differences exist |

**Interpretation**: The rankings are statistically significant - students genuinely care more about allergen labeling than vegan options.

---

## 3. Customer Segmentation

### Three Dietary Consciousness Segments

| Segment | Definition | Count | Percentage |
|---------|------------|-------|------------|
| **Dietary Indifferent** | Mean score < 2.0 | 117 | 72.7% |
| **Moderately Concerned** | Mean score 2.0-3.0 | 34 | 21.1% |
| **Dietary Conscious** | Mean score > 3.0 | 9 | 5.6% |

**Visualization**: `fig43_dietary_segments.png`

### Segment Profiles

| Metric | Indifferent | Moderate | Conscious |
|--------|-------------|----------|-----------|
| Mean Orders/Month | 2.4 | 2.6 | 2.4 |
| Mean Taste Importance | 4.5 | 4.4 | 4.3 |
| Mean Price Importance | 3.7 | 4.1 | 4.2 |
| % Prefer Local | 56.4% | 64.7% | 22.2% |

### Key Segment Insights

1. **Dietary Indifferent (73%)**: The majority. They order based on taste and price, not dietary options. This is the mainstream market.

2. **Moderately Concerned (21%)**: Care about allergen info but not specialized diets. They're more price-conscious and slightly more likely to prefer local.

3. **Dietary Conscious (6%)**: Small niche that actively seeks dietary accommodations. Surprisingly, they prefer chains (only 22% prefer local).

---

## 4. Correlation Analysis

### Dietary Factors vs Local/Chain Preference

| Dietary Factor | Correlation (r) | p-value | Significance |
|----------------|-----------------|---------|--------------|
| Plant-based protein | -0.162 | 0.042 | * |
| Reduced sodium | -0.147 | 0.064 | marginal |
| Vegan/dairy-free | -0.113 | 0.156 | NS |
| Gluten-free | -0.105 | 0.186 | NS |
| Allergen transparency | +0.027 | 0.739 | NS |

**Key Finding**: Students who value plant-based options are *less* likely to prefer local pizza places. This suggests chains may be perceived as better at dietary accommodations.

### Chi-Square Test: Dietary Consciousness vs Restaurant Choice

| Metric | Value |
|--------|-------|
| Chi-square | 1.251 |
| p-value | 0.535 |
| Cramér's V | 0.088 |
| Significance | **No** |

**Interpretation**: Dietary consciousness does NOT significantly predict whether students choose local or chain restaurants.

**Visualization**: `fig44_dietary_correlations.png`

---

## 5. Market Opportunity

### Dietary Comparison: Quality vs Dietary Factors

| Factor Category | Mean Importance |
|-----------------|-----------------|
| Pizza Quality Factors | 3.44/5 |
| Dietary Accommodation Factors | 1.67/5 |

**Quality factors are rated 106% higher than dietary factors.**

**Visualization**: `fig45_dietary_vs_quality.png`

### Unmet Dietary Needs

| Priority | Factor | Est. Students | Opportunity |
|----------|--------|---------------|-------------|
| 1 | Clear allergen labeling | ~47 (29%) | Low-cost signage/menu improvement |
| 2 | Allergen transparency | ~19 (12%) | Ingredient sourcing disclosure |
| 3 | Cross-contamination prevention | ~18 (11%) | Separate prep area messaging |
| 4 | Organic/non-GMO | ~16 (10%) | Premium positioning |
| 5 | Half-and-half option | ~21 (13%) | Easy menu addition |

### What Students Actively Seek (Q10)

Among students who actively seek dietary options (n=43):

| Option | Count | % of Seekers |
|--------|-------|--------------|
| Vegetarian | 17 | 39.5% |
| Other | 16 | 37.2% |
| Gluten-free | 9 | 20.9% |
| Lactose intolerant options | 6 | 14.0% |
| Vegan | 5 | 11.6% |

**Visualization**: `fig46_dietary_options_sought.png`

---

## 6. Demographic Insights

### Dietary Consciousness by Housing

| Housing Type | Mean Score | n |
|--------------|------------|---|
| Commuter student | 1.92 | 12 |
| Off campus | 1.67 | 79 |
| On campus | 1.63 | 67 |

**Insight**: Commuters show slightly higher dietary consciousness (possibly due to more meal planning).

### Dietary Consciousness by Year

| Year | Mean Score | n |
|------|------------|---|
| Junior | 1.80 | 40 |
| Freshman | 1.77 | 24 |
| Sophomore | 1.62 | 27 |
| Senior | 1.57 | 49 |

**Insight**: No significant differences by year. Dietary consciousness doesn't increase with age/experience.

---

## 7. Strategic Recommendations

### For a New Pizza Entrant

| Priority | Recommendation | Rationale |
|----------|----------------|-----------|
| **LOW** | Invest heavily in dietary accommodations | Only 27% care, and it doesn't drive local preference |
| **MEDIUM** | Offer clear allergen labeling | Highest-rated factor (2.52/5), low cost to implement |
| **MEDIUM** | Provide half-and-half pizza option | 13% high importance, easy operational addition |
| **LOW** | Vegan/gluten-free specialization | <2% rate highly important, niche market |

### What DOES Matter (Reference)

Based on overall findings, prioritize:
1. **Taste & Flavor** (4.44/5) - 94% rate highly important
2. **Price & Value** (3.80/5) - 67% rate highly important
3. **Convenience** (3.15/5) - 40% rate highly important

### Bottom Line Recommendation

> **Don't differentiate on dietary accommodations.** The market opportunity is too small. Instead:
> - Offer **basic allergen labeling** (low cost, addresses the #1 dietary concern)
> - Provide **half-and-half pizzas** for mixed groups
> - Focus investment on **taste, price, and speed**

---

## Appendix: Generated Visualizations

| Figure | Description |
|--------|-------------|
| `fig41_dietary_importance.png` | Dietary factor importance rankings (horizontal bar) |
| `fig42_dietary_distribution.png` | Full distribution of responses (stacked bar) |
| `fig43_dietary_segments.png` | Customer segmentation pie chart + profile comparison |
| `fig44_dietary_correlations.png` | Spearman correlation heatmap between dietary factors |
| `fig45_dietary_vs_quality.png` | Comparison: quality vs dietary importance |
| `fig46_dietary_options_sought.png` | Q10 responses - what students actively seek |

---

## Appendix: Data Sources

| Output File | Description |
|-------------|-------------|
| `outputs/dietary_importance_rankings.csv` | Full ranking data with statistics |
| `outputs/dietary_segments.csv` | Individual respondent segment assignments |

---

*Analysis scripts: `scripts/dietary_analysis.py`, `scripts/generate_dietary_visuals.py`*
