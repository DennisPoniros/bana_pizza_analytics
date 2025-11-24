# Competitive Analysis: Best Pizza Model

**Client**: Local Entrepreneurs Planning New Pizza Restaurant Near RIT
**Objective**: Data-driven go-to-market strategy based on student preferences
**Date**: November 2025

---

## Executive Summary

Using survey data from 161 RIT students, we developed a multi-component analytical model to answer: **"What makes a pizza place 'the best' and who is the primary competition?"**

### Key Findings

| Question | Answer |
|----------|--------|
| **Primary Competitor** | Domino's Pizza (Threat Score: 74.3/100) |
| **Top Local Competitor** | Joe's Brooklyn Pizza (Score: 61.2/100) |
| **#1 Success Factor** | Taste (14.1% weight, 94% rate highly important) |
| **Key Opportunity** | 28 Domino's customers prefer local but haven't switched |
| **Optimal Price Point** | $17-25 for 16" pizza |
| **Service Model** | Pickup-focused (71% preference) |

---

## Analytical Framework

We developed a **5-component model** to explain and predict pizza preferences:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BEST PIZZA ANALYTICAL MODEL                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. WEIGHTED IMPORTANCE MODEL                                       │
│     → Defines what "best" means (9 factors, normalized weights)     │
│                                                                     │
│  2. CUSTOMER SEGMENTATION                                           │
│     → Identifies distinct customer profiles (4 segments)            │
│                                                                     │
│  3. RESTAURANT PROFILING                                            │
│     → Maps competitors to customer segments                         │
│                                                                     │
│  4. REGRESSION ANALYSIS                                             │
│     → Predicts what drives loyalty and local preference             │
│                                                                     │
│  5. COMPETITIVE RANKING                                             │
│     → Composite threat score for strategic prioritization           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Weighted Importance Model

### What Defines "The Best" Pizza?

We calculated normalized importance weights from student ratings (Q5):

![Importance Weights](outputs/fig7_importance_weights.png)

| Rank | Factor | Mean (1-5) | Weight | % High Importance |
|------|--------|------------|--------|-------------------|
| 1 | **Taste & Flavor** | 4.44 | 14.1% | 94.4% |
| 2 | Balance & Ratios | 3.97 | 12.6% | 71.2% |
| 3 | Crust Excellence | 3.86 | 12.2% | 64.4% |
| 4 | Freshness & Temp | 3.83 | 12.1% | 65.6% |
| 5 | Price & Value | 3.80 | 12.0% | 66.9% |
| 6 | Ingredient Quality | 3.50 | 11.1% | 48.1% |
| 7 | Convenience | 3.15 | 10.0% | 40.0% |
| 8 | Appearance | 2.74 | 8.7% | 26.9% |
| 9 | Special Features | 2.29 | 7.2% | 13.8% |

### Model Equation

```
Best Pizza Score = 0.141(Taste) + 0.126(Balance) + 0.122(Crust) +
                   0.121(Freshness) + 0.120(Price) + 0.111(Ingredients) +
                   0.100(Convenience) + 0.087(Appearance) + 0.072(Special)
```

**Statistical Validation**: Friedman test confirms significant differences between factors (χ² = 435.85, p < 0.001)

### Strategic Implication

> **Taste is non-negotiable.** With 94% rating it highly important and 14.1% weight, a new entrant must deliver exceptional taste to compete. Special features and appearance are low-priority investments.

---

## Component 2: Customer Segmentation

Using hierarchical clustering on importance ratings, we identified **4 distinct customer segments**:

| Segment | Size | Local Pref | Top Priority | Top Restaurant |
|---------|------|------------|--------------|----------------|
| **Local Loyalists (High-Value)** | 37 (23%) | 93% | Taste, Balance | Domino's (35%) |
| **Local Loyalists (Quality)** | 33 (21%) | 91% | Taste, Price | Domino's (33%) |
| **Value Hunters** | 23 (14%) | 67% | Taste, Price, Crust | Domino's (35%) |
| **Balanced Buyers** | 67 (42%) | 80% | Taste, Price, Balance | Domino's (18%) |

### Key Insight

> **Even segments with 90%+ local preference choose Domino's.** This represents the core market opportunity—customers want local but default to chains.

---

## Component 3: Restaurant Competitive Profile

### Market Share & Customer Characteristics

| Restaurant | Share | Type | Loyalty (1-5) | Exp. Price | Customers Prefer Local |
|------------|-------|------|---------------|------------|------------------------|
| **Domino's Pizza** | 27.3% | Chain | 2.07 | $16 | 65% |
| Costco Pizza | 9.9% | Chain | 1.62 | $18 | 100% |
| Joe's Brooklyn | 9.3% | Local | 2.20 | $19 | 100% |
| Salvatore's | 7.5% | Local | 1.83 | $16 | 100% |
| Papa John's | 6.8% | Chain | 1.36 | $16 | 63% |
| Mark's Pizzeria | 5.6% | Local | 2.78 | $14 | 86% |
| Little Caesars | 5.6% | Chain | 2.67 | $14 | 100% |

### The Local-Chain Paradox

![Local Chain Paradox](outputs/fig9_local_chain_paradox.png)

**Left**: 57% state they prefer local
**Right**: Domino's (chain) captures 27% of the market

---

## Component 4: Regression Analysis

### What Predicts Customer Loyalty?

Multiple regression model (R² = 0.093, n = 159):

| Predictor | Coefficient | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| Order Frequency | +0.093 | **0.015*** | More orders → More loyal |
| Prefers Local (stated) | -0.441 | **0.029*** | Local-preferrers are less loyal to current choice |
| Price Importance | +0.184 | 0.121 | Price-sensitive = slightly more loyal |
| Taste Importance | -0.060 | 0.689 | Not significant |

**Key Finding**: Students who prefer local but currently choose a chain are **less loyal**—they're waiting for a better option.

### What Predicts Choosing LOCAL?

![Local Predictors](outputs/fig12_local_predictors.png)

| Factor | Local vs Chain Diff | p-value |
|--------|---------------------|---------|
| Stated "prefer local" | +37.5% | <0.001*** |
| Has transportation | +27.9% | 0.0004*** |
| Expected price | +$2.00 | 0.026* |
| Orders/month | +0.95 | 0.016* |
| Crust importance | +0.46 | 0.008** |

**Profile of Local Choosers**: Have cars, order more frequently, willing to pay more, care about crust quality.

---

## Component 5: Competitive Threat Ranking

### Composite Scoring Methodology

```
Threat Score = 0.30(Market Share) + 0.25(Loyalty) +
               0.25(Profile Match) + 0.20(Local Capture)
```

| Weight | Component | Rationale |
|--------|-----------|-----------|
| 30% | Market Share | Current customer base to defend |
| 25% | Loyalty | Retention strength (harder to poach) |
| 25% | Profile Match | Alignment with "ideal pizza" factors |
| 20% | Local Capture | Success attracting local-preferring customers |

### Competitive Ranking

![Competitive Ranking](outputs/fig8_competitive_ranking.png)

| Rank | Restaurant | Type | Threat Score |
|------|------------|------|--------------|
| **1** | **Domino's Pizza** | Chain | **74.3** |
| 2 | Joe's Brooklyn Pizza | Local | 61.2 |
| 3 | Little Caesars | Chain | 60.1 |
| 4 | Costco Pizza | Chain | 58.7 |
| 5 | Mark's Pizzeria | Local | 58.4 |

### Winner Identification

> **PRIMARY COMPETITOR: Domino's Pizza**
> - Threat Score: 74.3/100
> - Controls 27% of market despite 65% of their customers preferring local
> - Low loyalty (2.07/5) means customers are poachable

> **TOP LOCAL COMPETITOR: Joe's Brooklyn Pizza**
> - Threat Score: 61.2/100
> - 9.3% market share with strong crust/taste positioning
> - Model to study for local success factors

---

## Strategic Positioning Map

![Positioning Map](outputs/fig11_positioning_map.png)

The positioning map reveals four strategic quadrants:

| Quadrant | Description | Players |
|----------|-------------|---------|
| **Value Seekers** | Low price, high taste importance | Little Caesars, Mark's |
| **Premium Quality** | High price, high taste importance | Joe's Brooklyn, Pontillo's |
| **Budget Buyers** | Low price, moderate taste importance | Domino's, Papa John's |
| **Indifferent** | High price, moderate taste importance | Costco |

### Recommended Position for New Entrant

> **"Value Seekers" quadrant** - Match chain pricing (~$17) while delivering local quality. This directly attacks Domino's weakness.

---

## The Domino's Opportunity

![Domino's Opportunity](outputs/fig10_dominos_opportunity.png)

### Why Target Domino's Customers?

| Metric | Domino's | Others | Opportunity |
|--------|----------|--------|-------------|
| Loyalty Score | 2.07 | 2.16 | Low loyalty = easy to poach |
| "Prefer Local" | 65% | 43% | Want local, just need option |
| Expected Price | $16 | $18 | Price-sensitive segment |
| N Persuadable | **28** | - | Concrete target market |

### The 28-Customer Prize

28 Domino's customers (65% of 44) say they prefer local but currently choose Domino's. At 2.2 pizzas/month and $16/pizza, this represents:

```
28 customers × 2.2 orders/month × $16 = $985/month = $11,820/year
```

This is just the "persuadable" segment—total addressable market is much larger.

---

## Go-To-Market Recommendations

### 1. Product Strategy

| Priority | Factor | Target | Rationale |
|----------|--------|--------|-----------|
| **Critical** | Taste | Excellence | 94% high importance, 14.1% weight |
| **High** | Balance | Very Good | 71% high importance |
| **High** | Crust | Signature | Differentiator for local choosers |
| **Medium** | Freshness | Always Hot | 66% high importance |
| **Low** | Special Features | Skip | Only 14% care |

### 2. Pricing Strategy

| Tier | Price (16") | Positioning |
|------|-------------|-------------|
| **Entry** | $16-17 | Match Domino's, capture price-sensitive |
| **Standard** | $18-20 | Slight premium for quality |
| **Premium** | $22-25 | "Best pizza" positioning for enthusiasts |

**Sweet Spot**: $17 captures the Domino's customer while still signaling quality.

### 3. Service Model

| Channel | Investment | Rationale |
|---------|------------|-----------|
| **Pickup** | High | 71% preference, build efficient system |
| **In-store** | Medium | Ambiance for "local feel" |
| **Delivery** | Low | 24% use, offer but don't over-invest |

**Target pickup time**: 22 minutes (matches expectations)

### 4. Marketing Position

**Tagline Concept**: *"Local quality. Chain convenience."*

**Key Messages**:
- Fresh, handcrafted taste (addresses #1 factor)
- Fast pickup (addresses why students choose chains)
- Local business supporting RIT community
- Competitive pricing (removes barrier)

### 5. Location Considerations

| Factor | Finding | Recommendation |
|--------|---------|----------------|
| Transportation | 64% have cars | Location flexibility OK |
| Campus Proximity | 42% on-campus | Within delivery radius of dorms |
| Competition | Domino's nearby | Close enough to be alternative |

---

## Model Validation

### Statistical Rigor

| Component | Method | Validation |
|-----------|--------|------------|
| Importance Weights | Mean scores | Friedman test p < 0.001 |
| Segmentation | Hierarchical clustering | 4 distinct profiles |
| Predictors | Multiple regression | R² = 0.093, significant predictors |
| Group Comparisons | Independent t-tests | p < 0.05 for key differences |
| Threat Ranking | Composite scoring | Transparent, weighted formula |

### Limitations

1. **Self-reported preferences** may differ from actual behavior
2. **Sample limited to RIT students** - may not generalize
3. **No restaurant-level quality data** - inferred from customer values
4. **Cross-sectional data** - cannot assess trends

---

## Files Generated

| File | Purpose |
|------|---------|
| `competitive_model.py` | Main competitive analysis model |
| `regression_analysis.py` | Predictive regression models |
| `generate_competitive_visuals.py` | Visualization generation |
| `outputs/fig7_importance_weights.png` | Weighted importance model |
| `outputs/fig8_competitive_ranking.png` | Threat score rankings |
| `outputs/fig9_local_chain_paradox.png` | Stated vs actual preference |
| `outputs/fig10_dominos_opportunity.png` | Domino's customer analysis |
| `outputs/fig11_positioning_map.png` | Strategic positioning |
| `outputs/fig12_local_predictors.png` | Regression coefficients |

---

## Conclusion

The analytical model identifies **Domino's Pizza as the primary competitor** with a threat score of 74.3/100. Despite controlling 27% of the market, Domino's has critical vulnerabilities:

1. **Low customer loyalty** (2.07/5) - easily poached
2. **65% of customers prefer local** - cognitive dissonance
3. **Wins on convenience, not quality** - beatable on taste

The winning strategy for a new local entrant:

> **Match Domino's on convenience and price. Beat them on taste. Capture the 28+ customers who want local but haven't found their option.**

---

*Model developed using Python (pandas, scipy, matplotlib). Full methodology in [METHODOLOGY.md](METHODOLOGY.md).*
