# Final Model Report: Defining and Predicting "The Best Pizza"

**BANA255 Business Analytics Group Project**
**Version**: 1.0 | **Date**: December 2025

---

## Executive Summary

This report documents our **Final Analytical Model** for determining "the best pizza" available to RIT students. Our model integrates three complementary components:

1. **Weighted Importance Model**: Defines what "best" means by quantifying which pizza attributes matter most
2. **Behavioral Prediction Model**: Explains WHY students choose certain restaurants over others
3. **Competitive Ranking Model**: Identifies the primary competitor and market winner

**Final Winner**: **Domino's Pizza** (27.3% market share, statistically dominant)

**Key Strategic Insight**: 84% of students prefer local pizza, yet Domino's wins. The gap represents a $23.35/month/customer opportunity for a well-positioned local entrant.

---

## Table of Contents

1. [Model Design Philosophy](#1-model-design-philosophy)
2. [Component 1: Weighted Importance Model](#2-component-1-weighted-importance-model)
3. [Component 2: Behavioral Prediction Model](#3-component-2-behavioral-prediction-model)
4. [Component 3: Competitive Ranking Model](#4-component-3-competitive-ranking-model)
5. [Model Integration: The Complete Framework](#5-model-integration-the-complete-framework)
6. [Why Alternative Models Are Inappropriate](#6-why-alternative-models-are-inappropriate)
7. [Model Validation & Confidence](#7-model-validation--confidence)
8. [Final Model Output](#8-final-model-output)

---

## 1. Model Design Philosophy

### 1.1 The Core Challenge

"Best pizza" is **subjective and multidimensional**. A naive approach might simply count votes for each restaurant. However, this fails to answer the deeper question: **WHY do students perceive their favorite as "the best"?**

Our model design reflects three key principles:

| Principle | Implementation |
|-----------|----------------|
| **Comprehensiveness** | Capture all relevant dimensions (taste, price, convenience, etc.) |
| **Transparency** | Every weight and decision is data-driven and auditable |
| **Actionability** | Findings translate directly to business decisions |

### 1.2 Research Objectives Alignment

Per the assignment objectives, our model must:

| Objective | How Our Model Addresses It |
|-----------|---------------------------|
| Predict/explain WHY students perceive pizza as "the best" | Weighted Importance Model quantifies what matters; Behavioral Model explains choice drivers |
| Rank "the best" and name a winner | Competitive Ranking Model uses composite scoring to declare winner |
| Inform market entry strategy | Every model output maps to actionable recommendations |

### 1.3 Model Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL ANALYTICAL MODEL                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   COMPONENT 1                 COMPONENT 2                COMPONENT 3         │
│   ────────────               ────────────               ────────────         │
│   Weighted                   Behavioral                 Competitive          │
│   Importance                 Prediction                 Ranking              │
│   Model                      Model                      Model                │
│                                                                              │
│   "What does                 "Why do students           "Who wins            │
│    BEST mean?"               choose local               the market?"         │
│                              vs chain?"                                      │
│                                                                              │
│   ↓                          ↓                          ↓                    │
│   9 Quality Factors          28 Behavioral              4 Scoring            │
│   Normalized Weights         Features                   Dimensions           │
│                              71.1% Accuracy             Composite Score      │
│                                                                              │
│                         ┌─────────────────┐                                  │
│                         │   INTEGRATION   │                                  │
│                         │                 │                                  │
│                         │  Winner:        │                                  │
│                         │  DOMINO'S       │                                  │
│                         │  (27.3% share)  │                                  │
│                         │                 │                                  │
│                         │  Opportunity:   │                                  │
│                         │  LOCAL QUALITY  │                                  │
│                         │  AT CHAIN       │                                  │
│                         │  CONVENIENCE    │                                  │
│                         └─────────────────┘                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component 1: Weighted Importance Model

### 2.1 Purpose

**Define what "best pizza" means** by quantifying the relative importance of different pizza attributes.

### 2.2 Methodology

Students rated 9 pizza quality factors on a 5-point Likert scale (1=Not at all important, 5=Extremely important). We normalized these to create population-level weights that define the "ideal pizza."

**Mathematical Formulation:**

For each factor *i*:
- Raw Mean: μᵢ = (1/n) × Σ rating_j
- Normalized Weight: wᵢ = μᵢ / Σμ × 100%

### 2.3 Results: The "Best Pizza" Definition

| Rank | Factor | Mean (1-5) | Weight | % High Importance |
|------|--------|------------|--------|-------------------|
| 1 | **Taste & Flavor** | 4.44 | **14.1%** | 94.4% |
| 2 | Balance & Ratios | 3.97 | 12.6% | 71.2% |
| 3 | Crust Excellence | 3.86 | 12.2% | 64.4% |
| 4 | Freshness & Temp | 3.83 | 12.1% | 65.6% |
| 5 | Price & Value | 3.80 | 12.0% | 66.9% |
| 6 | Ingredient Quality | 3.50 | 11.1% | 48.1% |
| 7 | Convenience | 3.15 | 10.0% | 40.0% |
| 8 | Appearance | 2.74 | 8.7% | 26.9% |
| 9 | Special Features | 2.29 | 7.2% | 13.8% |

### 2.4 Interpretation

**"The Best Pizza" = Exceptional Taste + Balanced Execution + Fair Price**

The weights reveal a clear hierarchy:
- **Core Quality** (Taste + Balance + Crust + Freshness): 51% of total weight
- **Value Proposition** (Price + Ingredients): 23% of total weight
- **Convenience & Extras** (Convenience + Appearance + Special): 26% of total weight

### 2.5 Statistical Validation

- **Friedman Test**: χ² = 435.85, p < 0.001 → Factors are NOT equally important
- **Cronbach's Alpha**: α = 0.581 → Factors are distinct constructs (not unidimensional)
- **Effect Size**: Taste vs. Special Features: d = 2.1 (massive difference)

### 2.6 Why This Model Component is Appropriate

| Alternative Approach | Why It's Inferior |
|---------------------|-------------------|
| Equal weighting all factors | Ignores that taste matters more than appearance |
| Expert judgment | Subjective; our data is from actual customers |
| Single-factor model (e.g., taste only) | Misses price/convenience trade-offs |

---

## 3. Component 2: Behavioral Prediction Model

### 3.1 Purpose

**Explain WHY** students choose local vs. chain restaurants using observable behavioral features.

### 3.2 The Circular Feature Problem

**Critical Design Decision**: We deliberately **excluded** the "states prefer local" variable (Q17) from our predictive model.

**Why?** Using stated preference to predict actual choice is **tautological**:
- It answers: "Do people who say they prefer local, choose local?" (trivially yes)
- It provides **no actionable insight**
- It artificially inflates model accuracy

**Our Solution**: Predict local/chain choice using **only behavioral features** - what people DO, not what they SAY.

### 3.3 Feature Engineering: 28 Variables Across 6 Categories

| Category | Features | Rationale |
|----------|----------|-----------|
| **Quality Importance** (9) | imp_taste, imp_ingredients, imp_crust, imp_balance, imp_freshness, imp_appearance, imp_price, imp_convenience, imp_special | What customers VALUE → Product design |
| **Ordering Behavior** (4) | orders_per_month, prefers_pickup, prefers_delivery, orders_online | How customers INTERACT → Service model |
| **Time Preferences** (4) | expected_delivery, expected_pickup, willing_wait_delivery, willing_drive | Time expectations → Operations |
| **Price Sensitivity** (4) | expected_price, max_price, price_flexibility, price_over_location | Willingness to pay → Pricing strategy |
| **Demographics** (4) | age, has_transport, on_campus, year | Customer segments → Targeting |
| **Other** (3) | imp_variety, imp_foldability, deal_sensitivity | Secondary factors |

### 3.4 Model Ensemble Results

We trained 4 different algorithms to ensure robustness:

| Model | Test Accuracy | Cross-Val Mean | Key Strength |
|-------|---------------|----------------|--------------|
| Random Forest | 71.1% | 68.2% | Feature importance |
| Gradient Boosting | 68.9% | 67.1% | Non-linear patterns |
| Logistic Regression | 66.7% | 65.8% | Interpretability |
| Decision Tree | 64.4% | 62.3% | Decision rules |

**Consensus Accuracy: 71.1%** (without circular features)

### 3.5 Top Predictors of Local Choice

| Rank | Feature | Importance | Direction | Business Implication |
|------|---------|------------|-----------|---------------------|
| 1 | **Expected Pickup Time** | 0.717 | Local choosers expect longer | They plan for quality |
| 2 | **Prefers Pickup** | 0.630 | → Local | Pickup experience matters |
| 3 | **Max Price Willing** | 0.548 | Higher → Local | Local choosers pay more |
| 4 | **Orders Per Month** | 0.505 | Higher → Local | Frequent orderers choose local |
| 5 | **Price Flexibility** | 0.498 | Higher → Local | Willing to pay for quality |

### 3.6 Key Insight: The Local-Chain Paradox Explained

The model reveals **why** 84% say they prefer local but Domino's wins:

1. **Transportation as Barrier**: Students with cars are 2.7x more likely to choose local (44.7% vs. 16.4%, χ² = 11.43, p < 0.001)
2. **Convenience Constraints**: Pickup-preferring students choose local; delivery-dependent students default to chains
3. **Price Perception**: Chain customers expect lower prices ($16.50 vs. $18.20)

**Causal Pathway**: Stated Preference → BLOCKED BY → Convenience Constraints → Actual Choice

### 3.7 Why This Model Component is Appropriate

| Alternative Approach | Why It's Inferior |
|---------------------|-------------------|
| Include "states prefer local" | Circular logic; ~95% accuracy but no insight |
| Demographic-only model | Demographics explain <20% of variance |
| Simple logistic regression | Misses non-linear interactions |
| Satisfaction ratings | We don't have satisfaction data for all restaurants |

---

## 4. Component 3: Competitive Ranking Model

### 4.1 Purpose

**Identify the winner** and rank competitive threats for a new local entrant.

### 4.2 Scoring Methodology

We created a **composite threat score** combining 4 dimensions:

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| **Market Share** | 30% | Current dominance |
| **Customer Loyalty** | 25% | Retention strength |
| **Profile Match** | 25% | Alignment with "ideal pizza" preferences |
| **Local Capture** | 20% | Success with local-preferring customers |

**Formula:**
```
Threat Score = 0.30 × (Share_norm) + 0.25 × (Loyalty_norm) +
               0.25 × (ProfileMatch) + 0.20 × (LocalCapture)
```

### 4.3 Competitive Rankings

| Rank | Restaurant | Type | Share | Loyalty | Match | Local | **Composite** |
|------|------------|------|-------|---------|-------|-------|---------------|
| 1 | **Domino's Pizza** | Chain | 100.0 | 55.0 | 87.2 | 65.0 | **74.3** |
| 2 | Joe's Brooklyn | Local | 36.4 | 62.5 | 89.1 | 82.3 | 61.2 |
| 3 | Costco Pizza | Chain | 36.4 | 45.0 | 82.5 | 58.3 | 52.8 |
| 4 | Salvatore's | Local | 27.3 | 68.8 | 91.2 | 91.7 | 58.1 |
| 5 | Papa John's | Chain | 25.0 | 50.0 | 84.3 | 55.0 | 49.6 |

### 4.4 Winner Declaration

```
┌─────────────────────────────────────────────────────────────────┐
│                     WINNER: DOMINO'S PIZZA                       │
├─────────────────────────────────────────────────────────────────┤
│  Market Share:       27.3% (44 votes)                            │
│  Lead over #2:       17.4 percentage points                      │
│  Statistical Test:   Binomial p = 0.0012 (significant)           │
│  Threat Score:       74.3/100                                    │
├─────────────────────────────────────────────────────────────────┤
│  WHY DOMINO'S WINS:                                              │
│  ✓ Brand recognition and trust                                   │
│  ✓ Predictable quality (never great, never terrible)             │
│  ✓ Aggressive pricing ($7.99 deals)                              │
│  ✓ Fast, reliable delivery                                       │
│  ✓ Convenience (locations, hours, app)                           │
├─────────────────────────────────────────────────────────────────┤
│  DOMINO'S VULNERABILITY:                                         │
│  • 65% of their customers SAY they prefer local                  │
│  • That's 28+ "persuadable" customers                            │
│  • They win on convenience, NOT quality                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Why This Model Component is Appropriate

| Alternative Approach | Why It's Inferior |
|---------------------|-------------------|
| Simple vote count | Ignores loyalty, profile match, local capture |
| NPS/satisfaction ranking | We don't have NPS data |
| Revenue-based | No revenue data; share is proxy |
| Single-dimension (e.g., loyalty only) | Misses market share dominance |

---

## 5. Model Integration: The Complete Framework

### 5.1 How the Three Components Connect

```
COMPONENT 1                    COMPONENT 2                    COMPONENT 3
Weighted Importance    →       Behavioral Prediction   →      Competitive Ranking

"What is best?"               "Who chooses what?"            "Who wins?"

        │                              │                            │
        ▼                              ▼                            ▼
 Taste (14.1%)                 Transportation →              Domino's wins
 Balance (12.6%)               Local choice                  on convenience
 Crust (12.2%)
                               Pickup preference →           Local opportunity:
                               Local choice                  28 persuadables

                               Price flexibility →           Strategy:
                               Quality tolerance             Local quality +
                                                            Chain convenience
```

### 5.2 The Unified Insight

**The "best pizza" is defined by taste, but actual choice is determined by convenience.**

This creates the **Local-Chain Paradox**:
- Students know local pizza is better (84% prefer)
- But convenience constraints push them to chains (Domino's wins)
- The gap (46 pp) represents unmet demand

### 5.3 Business Application: The Scoring Formula

A new entrant can use our model to position themselves:

**Position Score = Σ (Factor_i × Weight_i) for all 9 factors**

| Scenario | Taste | Price | Convenience | ... | Position Score |
|----------|-------|-------|-------------|-----|----------------|
| Domino's (current) | 3.5 | 4.5 | 4.5 | ... | 3.57 |
| Joe's Brooklyn | 4.5 | 3.5 | 3.0 | ... | 3.86 |
| **New Entrant (target)** | **4.5** | **4.0** | **4.0** | ... | **4.12** |

The new entrant should target a position score of **4.0+** by matching local quality with chain convenience.

---

## 6. Why Alternative Models Are Inappropriate

### 6.1 Alternative 1: Simple Vote Count

**Approach**: Rank restaurants by number of "favorite" votes.

**Why Inappropriate**:
- ❌ Doesn't explain WHY students choose
- ❌ Ignores loyalty (repeat customers more valuable)
- ❌ Treats all votes equally (frequent orderers = casual)
- ❌ Provides no predictive power
- ❌ Can't inform product/pricing strategy

**Our model improvement**: Combines share with loyalty, profile match, and local capture for a richer competitive picture.

### 6.2 Alternative 2: Including Circular Features

**Approach**: Use "states prefer local" (Q17) to predict local choice.

**Why Inappropriate**:
- ❌ **Tautological**: Predicts that people do what they say they'll do
- ❌ **No insight**: Doesn't reveal what DRIVES preference
- ❌ **Inflated accuracy**: ~95% accuracy but meaningless
- ❌ **Not actionable**: Can't change stated preference; can change behavior

**Our model improvement**: Excludes circular features, revealing true behavioral predictors (transportation, pickup preference, price tolerance).

### 6.3 Alternative 3: Expert Judgment Weights

**Approach**: Have industry experts assign importance weights.

**Why Inappropriate**:
- ❌ **Subjective**: Different experts give different weights
- ❌ **Not customer-centric**: Experts may not reflect student preferences
- ❌ **Not replicable**: Can't audit or validate
- ❌ **May be outdated**: Expert knowledge may lag market

**Our model improvement**: Data-driven weights from actual customer ratings.

### 6.4 Alternative 4: Single-Factor Model (e.g., Taste Only)

**Approach**: Assume "best" = best taste and rank accordingly.

**Why Inappropriate**:
- ❌ **Ignores trade-offs**: Students balance taste vs. price vs. convenience
- ❌ **Doesn't match behavior**: Domino's wins despite lower taste perception
- ❌ **Limited strategy**: Only informs product, not pricing/service

**Our model improvement**: 9-factor weighted model captures the full decision landscape.

### 6.5 Alternative 5: Demographic-Only Prediction

**Approach**: Predict choice using only demographics (age, gender, housing).

**Why Inappropriate**:
- ❌ **Low explanatory power**: Demographics explain <20% of variance
- ❌ **Not actionable**: Can't change customer demographics
- ❌ **Misses behavior**: How they order matters more than who they are

**Our model improvement**: 28 features including behavior, preferences, and demographics.

### 6.6 Alternative 6: Satisfaction-Based Ranking

**Approach**: Rank restaurants by customer satisfaction scores.

**Why Inappropriate**:
- ❌ **Data unavailable**: We don't have satisfaction ratings for all restaurants
- ❌ **Selection bias**: Only have data from people who chose each restaurant
- ❌ **Can't compare non-customers**: How satisfied would Domino's customers be at Joe's?

**Our model improvement**: Use importance × share × loyalty as a proxy for competitive strength.

---

## 7. Model Validation & Confidence

### 7.1 Statistical Validation Summary

| Component | Test | Result | Interpretation |
|-----------|------|--------|----------------|
| Importance Model | Friedman χ² | 435.85, p < 0.001 | Factors differ significantly |
| Importance Model | Cronbach's α | 0.581 | Factors are distinct constructs |
| Behavioral Model | Test Accuracy | 71.1% | Better than random (50%) |
| Behavioral Model | Cross-Val | 68.2% ± 4.1% | Stable across folds |
| Winner Declaration | Binomial test | p = 0.0012 | Domino's lead is significant |
| Local Preference | Chi-square | χ² = 11.43, p < 0.001 | Transportation predicts choice |

### 7.2 Confidence Intervals for Key Findings

| Finding | Estimate | 95% CI | Interpretation |
|---------|----------|--------|----------------|
| Local preference | 84.1% | [77.8%, 89.5%] | Strong majority prefer local |
| Domino's share | 27.3% | [21.2%, 34.2%] | Clear market leader |
| Taste importance | 4.44/5 | [4.35, 4.53] | Near-ceiling priority |
| Price flexibility | $7.93 | [$6.81, $9.05] | ~46% premium tolerance |

### 7.3 Sensitivity Analysis

**Scenario 1**: What if Domino's share is overstated by 20%?
- Adjusted share: 21.8%
- Result: Still #1 by 12+ points → **Finding robust**

**Scenario 2**: What if local preference is overstated by 30%?
- Adjusted preference: 58.9%
- Result: Still majority prefer local → **Finding robust**

**Scenario 3**: What if ML accuracy drops to 60%?
- Result: Still better than random; key predictors unchanged → **Finding robust**

### 7.4 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Convenience sample | May not represent all RIT students | Large n=161 reduces bias |
| Self-reported data | Stated ≠ revealed preference | Used behavioral features |
| Single time point | Nov 2025 only | Noted as limitation |
| RIT-specific | May not generalize | Appropriate for local strategy |

---

## 8. Final Model Output

### 8.1 The Winner

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    PRIMARY COMPETITOR:  DOMINO'S PIZZA                                    ║
║                                                                           ║
║    Market Share:        27.3%                                             ║
║    Threat Score:        74.3/100                                          ║
║    Statistical Sig.:    p = 0.0012                                        ║
║                                                                           ║
║    Wins Because:        Convenience + Price + Brand Trust                 ║
║    Vulnerable Because:  65% of customers prefer local                     ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 8.2 What "Best Pizza" Means

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    "THE BEST PIZZA" = Weighted Sum of 9 Factors                          ║
║                                                                           ║
║    Taste & Flavor         14.1%  ████████████████                         ║
║    Balance & Ratios       12.6%  ██████████████                           ║
║    Crust Excellence       12.2%  █████████████                            ║
║    Freshness & Temp       12.1%  █████████████                            ║
║    Price & Value          12.0%  █████████████                            ║
║    Ingredient Quality     11.1%  ████████████                             ║
║    Convenience            10.0%  ███████████                              ║
║    Appearance              8.7%  █████████                                ║
║    Special Features        7.2%  ████████                                 ║
║                                                                           ║
║    CORE QUALITY: 51%  |  VALUE: 23%  |  CONVENIENCE: 26%                 ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 8.3 Why Students Choose (Behavioral Drivers)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    TOP PREDICTORS OF LOCAL CHOICE (Behavioral Model, 71.1% Acc.)          ║
║                                                                           ║
║    1. Expected Pickup Time     → Local choosers plan for quality          ║
║    2. Prefers Pickup           → Pickup culture enables local             ║
║    3. Max Price Willing        → Local choosers pay premium               ║
║    4. Has Transportation       → Cars enable local access                 ║
║    5. Orders Frequently        → Pizza enthusiasts choose local           ║
║                                                                           ║
║    KEY INSIGHT: Convenience constraints block stated preference           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 8.4 Strategic Recommendation

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    NEW ENTRANT POSITIONING: "LOCAL QUALITY AT CHAIN CONVENIENCE"          ║
║                                                                           ║
║    ┌─────────────┬─────────────────────────────────────────────┐         ║
║    │ PRODUCT     │ Exceptional taste (94% rate highly important)│         ║
║    │ PRICE       │ $18-20 (optimal revenue point)               │         ║
║    │ SERVICE     │ Fast pickup (<22 min)                        │         ║
║    │ TARGET      │ Students with transportation (2.7x lift)     │         ║
║    │ SIDES       │ Garlic knots + Wings (65%, 53% interest)     │         ║
║    └─────────────┴─────────────────────────────────────────────┘         ║
║                                                                           ║
║    MARKET OPPORTUNITY:                                                    ║
║    • 28+ persuadable customers at Domino's                               ║
║    • 46 pp gap between preference and behavior                           ║
║    • $23.35/month/customer revenue potential                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Appendix: Model Reproducibility

All analyses can be reproduced using the following scripts:

| Script | Component | Output |
|--------|-----------|--------|
| `scripts/competitive_model.py` | Weighted Importance + Segmentation | Importance weights, threat scores |
| `scripts/ensemble_model.py` | Behavioral Prediction | ML accuracy, feature importance |
| `scripts/strategic_analysis.py` | Winner + Causality | Winner declaration, causal tests |
| `scripts/advanced_statistics.py` | Validation | PCA, clustering, mediation |

**Random Seed**: 42 (for reproducibility)
**Software**: Python 3.11, pandas, scipy, scikit-learn, matplotlib, seaborn

---

*This report accompanies the BANA255 Business Analytics Group Project presentation.*
