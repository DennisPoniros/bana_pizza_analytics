# BANA255 Best Pizza Analytics

**Research Question**: What is the best pizza available to RIT students?

**Client**: Local entrepreneurs planning a new pizza restaurant near RIT campus

**Objective**: Provide data-driven insights and go-to-market strategy

**Version**: 8.0 (December 2025) | **52 Visualizations** | **8 Analysis Components**

---

## Project Overview

This repository contains survey data from 161 RIT students and a comprehensive analytics pipeline with **8 major analytical components** to answer:

1. **What defines "the best" pizza?** (Weighted importance model)
2. **Who is the primary competition?** (Competitive threat ranking)
3. **What predicts customer choice?** (Machine learning model)
4. **How should a new local entrant position itself?** (Go-to-market strategy)
5. **What side orders drive revenue?** (Side order analysis)
6. **What dietary accommodations matter?** (Dietary preferences analysis)
7. **How reliable is our data?** (Data quality assessment)

---

## Key Findings

| Question | Answer |
|----------|--------|
| **#1 Quality Factor** | Taste (94% rate highly important) |
| **Primary Competitor** | Domino's Pizza (27% market share) |
| **Top Local Competitor** | Joe's Brooklyn Pizza (9% share) |
| **Optimal Price Point** | $17-25 for 16" pizza |
| **Service Model** | Pickup-focused (71% preference) |
| **ML Prediction Accuracy** | 71.1% (without circular features) |
| **Top Side Orders** | Garlic knots (65%), Wings (53%) |
| **Side Spend/Order** | $9.38 average |
| **Data Quality Score** | 92/100 (Excellent) |

### The Local-Chain Paradox

84% of students **say** they prefer local pizza, yet Domino's (a chain) captures 27% of the market. This represents a key opportunity for a well-positioned local entrant.

---

## Repository Structure

```
bana_pizza_analytics/
│
├── README.md                    # This file - project overview
├── FINDINGS_REPORT.md           # Master findings document (comprehensive)
│
├── reports/                     # Detailed analysis reports
│   ├── EXECUTIVE_SUMMARY.md     # Visual summary for quick review
│   ├── COMPETITIVE_ANALYSIS.md  # Full competitive model & strategy
│   ├── ML_MODEL_REPORT.md       # Machine learning model details
│   ├── STRATEGIC_ANALYSIS.md    # Winner declaration & causal analysis
│   ├── DIETARY_ANALYSIS.md      # Dietary accommodation findings
│   └── METHODOLOGY.md           # Statistical methods & references
│
├── scripts/                     # Analysis code (15 scripts, 7,500+ lines)
│   ├── pizza_analysis.py        # Descriptive statistics
│   ├── competitive_model.py     # Competitive scoring model
│   ├── regression_analysis.py   # Regression analysis
│   ├── ensemble_model.py        # ML ensemble model
│   ├── strategic_analysis.py    # Winner & causal analysis
│   ├── advanced_statistics.py   # PCA, clustering, mediation, etc.
│   ├── dietary_analysis.py      # Dietary preferences analysis
│   ├── side_order_analysis.py   # Side order revenue analysis (NEW)
│   ├── data_quality_analysis.py # Data quality assessment (NEW)
│   └── generate_*.py            # Visualization generators
│
└── outputs/                     # Generated figures and data
    ├── fig1-6: Descriptive analysis
    ├── fig7-12: Competitive model
    ├── fig13-17: Machine learning
    ├── fig18-28: Advanced presentation
    ├── fig29-40: Advanced statistics
    ├── fig41-46: Dietary analysis
    ├── fig47-52: Side order analysis (NEW)
    └── *.csv: Exported data tables
```

---

## Analysis Components

### 1. Descriptive Analysis (`pizza_analysis.py`)
- Local vs chain preference distribution
- Customer visit frequency by restaurant
- Quality characteristic importance rankings
- Delivery vs pickup preferences

### 2. Competitive Model (`competitive_model.py`)
- Weighted importance model (defines "best pizza")
- Customer segmentation (4 distinct profiles)
- Composite threat score ranking

### 3. Regression Analysis (`regression_analysis.py`)
- Multiple regression for loyalty predictors
- Group comparisons (local vs chain choosers)

### 4. Machine Learning Model (`ensemble_model.py`)
- Random Forest, Gradient Boosting, Logistic Regression
- 71.1% accuracy (circular features excluded)
- Feature importance consensus ranking

### 5. Advanced Statistics (`advanced_statistics.py`)
- PCA (4 components, 65% variance)
- K-Means clustering validation
- Van Westendorp price sensitivity
- Mediation analysis, LDA, propensity scores

### 6. Dietary Analysis (`dietary_analysis.py`)
- 11 dietary factors analyzed
- Customer segmentation by dietary consciousness
- Finding: Dietary accommodations are LOW priority (avg 1.67/5)

### 7. Side Order Analysis (`side_order_analysis.py`) *NEW*
- 10 side items ranked by popularity
- Revenue opportunity assessment ($9.38 avg spend)
- Finding: Garlic knots (65%) and Wings (53%) are must-haves

### 8. Data Quality Analysis (`data_quality_analysis.py`) *NEW*
- Missing data assessment (98.6% complete)
- Outlier detection
- Power analysis (adequate for medium+ effects)
- Quality score: 92/100 (Excellent)

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl

# Run all analyses (from repository root)
python scripts/pizza_analysis.py           # Descriptive stats
python scripts/competitive_model.py        # Competitive analysis
python scripts/regression_analysis.py      # Regression models
python scripts/ensemble_model.py           # ML model
python scripts/advanced_statistics.py      # Advanced stats
python scripts/dietary_analysis.py         # Dietary analysis
python scripts/side_order_analysis.py      # Side order analysis
python scripts/data_quality_analysis.py    # Data quality

# Generate all visualizations
python scripts/generate_summary.py                 # fig1-6
python scripts/generate_competitive_visuals.py    # fig7-12
python scripts/generate_ml_visuals.py             # fig13-17
python scripts/generate_advanced_visuals.py       # fig18-28
python scripts/generate_advanced_stats_visuals.py # fig29-40
python scripts/generate_dietary_visuals.py        # fig41-46
python scripts/generate_side_order_visuals.py     # fig47-52
```

---

## Key Visualizations

| Figure | Description | Location |
|--------|-------------|----------|
| fig1 | Local vs Chain Preference | `outputs/` |
| fig2 | Top 10 Pizza Places | `outputs/` |
| fig3 | Quality Importance Ratings | `outputs/` |
| fig7 | Weighted Importance Model | `outputs/` |
| fig8 | Competitive Threat Ranking | `outputs/` |
| fig11 | Strategic Positioning Map | `outputs/` |
| fig13 | ML Feature Importance | `outputs/` |
| fig35 | Van Westendorp Price Sensitivity | `outputs/` |
| fig47 | Side Order Popularity | `outputs/` |
| fig52 | Revenue Opportunity Matrix | `outputs/` |

---

## Statistical Methods

| Method | Purpose | Key Finding |
|--------|---------|-------------|
| Chi-square | Preference testing | p < 0.001 (local preferred) |
| Friedman Test | Importance rankings | Taste > Price > Convenience |
| Random Forest | Behavioral prediction | 71.1% accuracy |
| PCA | Dimension reduction | 4 components, 65% variance |
| Van Westendorp | Price sensitivity | Optimal: $20 |
| Propensity Score | Causal inference | Loyalty diff disappears |
| K-Means | Segmentation | 3 side-order customer types |

See `reports/METHODOLOGY.md` for full statistical justification.

---

## Business Recommendations

For a new local pizza entrant near RIT:

1. **Product**: Exceptional taste (non-negotiable), quality crust
2. **Price**: $18-20 for 16" pizza (optimal revenue point)
3. **Service**: Fast pickup-focused model (<22 minutes)
4. **Position**: "Local quality at chain convenience"
5. **Target**: Students with transportation (2.7x more likely to choose local)
6. **Sides**: Must offer garlic knots and wings ($10-12 range)
7. **Skip**: Dietary specialization (low demand)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-24 | Initial descriptive analysis |
| 2.0 | 2025-11-24 | Competitive model |
| 3.0 | 2025-11-24 | ML ensemble model |
| 4.0 | 2025-11-24 | Strategic analysis |
| 5.0 | 2025-11-25 | Advanced visualizations |
| 6.0 | 2025-11-25 | Advanced statistics |
| 7.0 | 2025-11-28 | Dietary analysis |
| 8.0 | 2025-12-01 | **Side order analysis, Data quality** |

---

## Authors

BANA255 Analytics Team, Fall 2025

---

## License

Academic use only. Data collected under IRB-approved survey protocol.
