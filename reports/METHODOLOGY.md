# Statistical Methodology Report

**Study**: BANA255 Pizza Survey Analysis
**Document Purpose**: Detailed explanation and justification of statistical methods
**Last Updated**: November 24, 2025

---

## Table of Contents
1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Statistical Tests by Research Question](#statistical-tests-by-research-question)
4. [Assumptions & Diagnostics](#assumptions--diagnostics)
5. [Effect Sizes & Practical Significance](#effect-sizes--practical-significance)
6. [References](#references)

---

## Overview

### Statistical Framework
This analysis employs both **parametric** and **non-parametric** statistical methods depending on data characteristics:

| Data Type | Scale | Preferred Methods |
|-----------|-------|-------------------|
| Categorical (nominal) | Unordered categories | Chi-square, Binomial tests |
| Ordinal | Ordered categories (Likert) | Non-parametric (Wilcoxon, Friedman) |
| Continuous | Interval/ratio | t-tests, ANOVA |

### Significance Level
All hypothesis tests use **α = 0.05** (95% confidence level), the conventional threshold in social science research. Results with p < 0.05 are considered statistically significant.

### Software
- **Python 3.11** with pandas (data manipulation) and scipy.stats (statistical tests)
- All tests use two-tailed alternatives unless otherwise specified

---

## Data Preparation

### Sample Selection
```
Total responses collected: 164
Excluded (no consent): 3
Final analysis sample: 161
```

**Justification**: Only consented participants (Q2 = "Yes") are included per ethical research standards. The 98.2% consent rate suggests minimal selection bias from exclusions.

### Missing Data Handling
- **Approach**: Pairwise deletion (analyze all available data for each test)
- **Rationale**: Preserves maximum sample size; survey completion rate was high (>95% on most questions)
- **Alternative considered**: Listwise deletion would reduce sample unnecessarily

### Variable Transformations
Likert scale responses were converted to numeric scores:

| Response | Numeric Code |
|----------|--------------|
| Not at all important | 1 |
| Slightly important | 2 |
| Moderately important | 3 |
| Very important | 4 |
| Extremely important | 5 |

**Justification**: While Likert data is technically ordinal, treating 5-point scales as interval data is widely accepted in survey research when:
- Scale points are evenly spaced conceptually
- Sample size is adequate (n > 30)
- Parametric and non-parametric tests yield similar conclusions

> "For Likert scales with five or more categories, treating the data as continuous typically yields valid results" (Norman, 2010).

---

## Statistical Tests by Research Question

### RQ1: Local vs Chain Preference

#### Test 1: Chi-Square Goodness of Fit

**Purpose**: Determine if the distribution of preferences (Local/Chain/Unsure) differs from equal distribution.

**Formula**:
```
χ² = Σ [(Observed - Expected)² / Expected]
```

**Application**:
| Category | Observed | Expected (H₀) |
|----------|----------|---------------|
| Local | 90 | 53 |
| Chain | 17 | 53 |
| Unsure | 52 | 53 |

**Result**: χ² = 50.30, df = 2, p < 0.001

**Why this test?**
- Single categorical variable with 3+ categories
- Testing against theoretical distribution (equal preference)
- Sample size adequate: all expected frequencies > 5

**Assumptions**:
- [x] Independence of observations (each respondent answers once)
- [x] Expected frequencies ≥ 5 in all cells

> Reference: Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. *Philosophical Magazine*, 50, 157-175.

---

#### Test 2: Binomial Test

**Purpose**: Test if the proportion preferring "Local" differs from 50% (when excluding "Unsure" responses).

**Application**:
- Successes (Local): 90
- Trials (Local + Chain): 107
- H₀: p = 0.50 (equal preference)
- H₁: p ≠ 0.50

**Result**: p < 0.001, 95% CI [0.758, 0.905]

**Why this test?**
- Binary outcome (Local vs Chain)
- Testing against known proportion
- Exact test (no approximation needed)

**Interpretation**: With 95% confidence, between 75.8% and 90.5% of students who express a clear preference favor local establishments.

> Reference: Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404-413.

---

### RQ2: Visit Frequency by Restaurant

#### Test: One-Way ANOVA

**Purpose**: Determine if mean pizza ordering frequency differs across favorite restaurants.

**Formula**:
```
F = (Between-group variance) / (Within-group variance)
   = MSB / MSW
```

**Application**:
- Groups: 18 restaurants
- Dependent variable: Monthly order frequency (Q4)
- Groups with n < 3 excluded from ANOVA

**Result**: F = 1.307, p = 0.211

**Why this test?**
- Comparing means across multiple independent groups
- Continuous dependent variable
- Parametric alternative to Kruskal-Wallis

**Assumptions**:
- [x] Independence: Different respondents in each group
- [~] Normality: Violated for some groups, but ANOVA is robust with n > 30 total
- [~] Homogeneity of variance: Some variation exists; results should be interpreted cautiously

**Robustness Note**: ANOVA is robust to normality violations when:
- Total sample size is large (n = 161)
- Group sizes are reasonably similar
- No extreme outliers

> Reference: Glass, G. V., Peckham, P. D., & Sanders, J. R. (1972). Consequences of failure to meet assumptions underlying the fixed effects analyses of variance and covariance. *Review of Educational Research*, 42(3), 237-288.

---

### RQ3: Pizza Characteristic Importance

#### Test: Friedman Test

**Purpose**: Determine if importance ratings differ across the 9 pizza characteristics (repeated measures on ordinal data).

**Formula**:
```
χ²F = [12 / (nk(k+1))] × Σ(Rj²) - 3n(k+1)

where:
  n = number of subjects
  k = number of conditions
  Rj = sum of ranks for condition j
```

**Application**:
- Subjects: 161 respondents
- Conditions: 9 pizza characteristics (Q5_1 through Q5_9)
- Each respondent rated all 9 characteristics

**Result**: χ² = 435.85, df = 8, p < 0.001

**Why this test?**
- Repeated measures design (same people rate multiple items)
- Ordinal data (Likert scales)
- Non-parametric: No normality assumption required
- Alternative to repeated-measures ANOVA for ordinal data

**Assumptions**:
- [x] Random sample from population
- [x] Observations within each block can be ranked
- [x] Blocks are mutually independent

**Post-hoc consideration**: With significant Friedman result, pairwise Wilcoxon tests (see RQ5) identify which specific pairs differ.

> Reference: Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701.

---

### RQ4: Time Expectations for "Best" Pizza

#### Test: Paired t-test

**Purpose**: Determine if respondents are willing to wait significantly longer for "the best" pizza compared to their standard expectation.

**Formula**:
```
t = (mean difference) / (SE of difference)
  = d̄ / (sd / √n)
```

**Application** (Delivery):
- Pair 1: Expected delivery time (Q14_1)
- Pair 2: Willing to wait for best (Q15_1)
- Same respondents answered both questions

| Comparison | Mean Diff | t-statistic | p-value |
|------------|-----------|-------------|---------|
| Delivery | +13.6 min | significant | < 0.001 |
| Pickup | +6.5 min | significant | < 0.001 |

**Why this test?**
- Comparing two related measurements (same person, different conditions)
- Continuous outcome (minutes)
- Paired design controls for individual differences

**Assumptions**:
- [x] Paired observations (same respondent)
- [x] Continuous measurement scale
- [~] Normal distribution of differences: Approximately met with n > 100; t-test is robust

**Effect Size** (Cohen's d for paired samples):
```
d = mean difference / SD of differences
```
- Delivery: d ≈ 0.7 (medium-large effect)
- Pickup: d ≈ 0.4 (small-medium effect)

> Reference: Student [Gosset, W. S.]. (1908). The probable error of a mean. *Biometrika*, 6(1), 1-25.

---

### RQ5: Factor Importance Comparison

#### Test: Wilcoxon Signed-Rank Test

**Purpose**: Pairwise comparison of importance ratings between specific factors (e.g., Taste vs Price).

**Procedure**:
1. Calculate differences between paired observations
2. Rank absolute differences
3. Sum ranks of positive and negative differences
4. Test statistic based on smaller sum

**Application**:
| Comparison | Result | p-value | Winner |
|------------|--------|---------|--------|
| Taste vs Price | Taste > Price | < 0.001 | Taste |
| Taste vs Convenience | Taste > Convenience | < 0.001 | Taste |
| Price vs Convenience | Price > Convenience | < 0.001 | Price |

**Why this test?**
- Ordinal data (Likert scales)
- Paired observations (same person rates both factors)
- Non-parametric: No normality assumption
- More appropriate than paired t-test for ordinal data

**Assumptions**:
- [x] Paired observations
- [x] Ordinal or continuous scale
- [x] Symmetric distribution of differences (approximately met)

**Multiple Comparisons Note**: With 3 pairwise tests, consider Bonferroni correction:
- Adjusted α = 0.05 / 3 = 0.017
- All results remain significant even with correction

> Reference: Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.

---

## Assumptions & Diagnostics

### Summary of Assumption Checks

| Test | Key Assumptions | Status |
|------|-----------------|--------|
| Chi-square | Expected freq ≥ 5 | Met |
| Binomial | Independence | Met |
| ANOVA | Normality, homogeneity | Partially met; robust |
| Friedman | Rankable observations | Met |
| Paired t-test | Normal differences | Approximately met |
| Wilcoxon | Symmetric differences | Met |

### When Assumptions Are Violated

**Strategy employed**: Use non-parametric alternatives when parametric assumptions are questionable:
- Friedman instead of repeated-measures ANOVA
- Wilcoxon instead of paired t-test for Likert data
- Report both when results converge (increases confidence)

---

## Effect Sizes & Practical Significance

Statistical significance (p < 0.05) indicates an effect is unlikely due to chance, but **effect size** indicates practical importance.

### Effect Size Interpretations Used

| Measure | Small | Medium | Large |
|---------|-------|--------|-------|
| Cohen's d | 0.2 | 0.5 | 0.8 |
| Cramér's V | 0.1 | 0.3 | 0.5 |
| η² (eta-squared) | 0.01 | 0.06 | 0.14 |

### Key Effect Sizes in This Study

| Finding | Effect Size | Interpretation |
|---------|-------------|----------------|
| Local vs Chain preference | 84% vs 16% | Large practical difference |
| Taste importance | Mean 4.44/5.00 | Near ceiling effect |
| Extra wait for "best" (delivery) | +14 min (~40% increase) | Meaningful behavioral difference |
| Extra drive for "best" (pickup) | +6.5 min (~29% increase) | Moderate behavioral difference |

---

## References

### Statistical Methods

1. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
   - Standard reference for effect size interpretation

2. Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). SAGE Publications.
   - Comprehensive guide to statistical test selection

3. Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701.
   - Original Friedman test paper

4. Glass, G. V., Peckham, P. D., & Sanders, J. R. (1972). Consequences of failure to meet assumptions underlying the fixed effects analyses of variance and covariance. *Review of Educational Research*, 42(3), 237-288.
   - ANOVA robustness to assumption violations

5. Norman, G. (2010). Likert scales, levels of measurement and the "laws" of statistics. *Advances in Health Sciences Education*, 15(5), 625-632.
   - Justification for treating Likert data as interval

6. Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.
   - Original Wilcoxon signed-rank test paper

### Survey Methodology

7. Dillman, D. A., Smyth, J. D., & Christian, L. M. (2014). *Internet, Phone, Mail, and Mixed-Mode Surveys: The Tailored Design Method* (4th ed.). Wiley.
   - Survey design best practices

8. Krosnick, J. A., & Presser, S. (2010). Question and questionnaire design. In P. V. Marsden & J. D. Wright (Eds.), *Handbook of Survey Research* (2nd ed., pp. 263-313). Emerald.
   - Likert scale design considerations

### Online Resources

9. UCLA Statistical Consulting: https://stats.oarc.ucla.edu/
   - Test selection guides and assumption checking

10. SciPy Documentation: https://docs.scipy.org/doc/scipy/reference/stats.html
    - Implementation details for statistical functions used

---

## Appendix: Test Selection Decision Tree

```
Is the data categorical or continuous?
│
├─► Categorical
│   │
│   ├─► One variable → Chi-square goodness of fit
│   │
│   └─► Two variables → Chi-square test of independence
│
└─► Continuous/Ordinal
    │
    ├─► Comparing 2 groups
    │   │
    │   ├─► Independent → t-test / Mann-Whitney U
    │   │
    │   └─► Paired → Paired t-test / Wilcoxon signed-rank
    │
    └─► Comparing 3+ groups
        │
        ├─► Independent → ANOVA / Kruskal-Wallis
        │
        └─► Repeated measures → Repeated ANOVA / Friedman
```

---

*This methodology document accompanies the Findings Report and pizza_analysis.py script.*
