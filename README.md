# BANA255 Best Pizza Analytics

**Research Question**: What is the best pizza available to RIT students?

**Client**: Local entrepreneurs planning a new pizza restaurant near RIT campus

**Objective**: Provide data-driven insights and go-to-market strategy

---

## Project Overview

This repository contains survey data from 161 RIT students and a comprehensive analytics pipeline to answer:

1. **What defines "the best" pizza?** (Weighted importance model)
2. **Who is the primary competition?** (Competitive threat ranking)
3. **What predicts customer choice?** (Machine learning model)
4. **How should a new local entrant position itself?** (Go-to-market strategy)

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

### The Local-Chain Paradox

84% of students **say** they prefer local pizza, yet Domino's (a chain) captures 27% of the market. This represents a key opportunity for a well-positioned local entrant.

---

## Repository Structure

```
bana_pizza_analytics/
│
├── README.md                    # This file - project overview
├── FINDINGS_REPORT.md           # Master findings document with roadmap
│
├── data/
│   ├── BANA255+Best+Pizza+F25_November+17_2C+2025_11.42.xlsx  # Survey responses
│   └── BANA255_Best_Pizza_F25-2.docx                          # Survey instrument
│
├── reports/                     # Detailed analysis reports
│   ├── EXECUTIVE_SUMMARY.md     # Visual summary for quick review
│   ├── COMPETITIVE_ANALYSIS.md  # Full competitive model & strategy
│   ├── ML_MODEL_REPORT.md       # Machine learning model details
│   └── METHODOLOGY.md           # Statistical methods & references
│
├── scripts/                     # Analysis code
│   ├── pizza_analysis.py        # Descriptive statistics
│   ├── competitive_model.py     # Competitive scoring model
│   ├── regression_analysis.py   # Regression analysis
│   ├── ensemble_model.py        # ML ensemble model
│   ├── generate_summary.py      # Visualization (fig1-6)
│   ├── generate_competitive_visuals.py  # Visualization (fig7-12)
│   └── generate_ml_visuals.py   # Visualization (fig13-17)
│
└── outputs/                     # Generated figures and data
    ├── fig1-6: Descriptive analysis
    ├── fig7-12: Competitive model
    ├── fig13-17: Machine learning
    └── *.csv: Exported data tables
```

---

## Analysis Components

### 1. Descriptive Analysis (`pizza_analysis.py`)
**Purpose**: Answer initial research questions about pizza preferences

- Local vs chain preference distribution
- Customer visit frequency by restaurant
- Quality characteristic importance rankings
- Delivery vs pickup preferences
- Key decision factors (taste, price, convenience, variety)

### 2. Competitive Model (`competitive_model.py`)
**Purpose**: Identify and rank competitors for strategic positioning

- Weighted importance model (defines "best pizza")
- Customer segmentation (4 distinct profiles)
- Restaurant competitive profiling
- Composite threat score ranking
- Go-to-market recommendations

### 3. Regression Analysis (`regression_analysis.py`)
**Purpose**: Explain what predicts loyalty and local preference

- Multiple regression for loyalty predictors
- Group comparisons (local vs chain choosers)
- Domino's customer profile analysis
- Statistical significance testing

### 4. Machine Learning Model (`ensemble_model.py`)
**Purpose**: Predict local vs chain choice from behavioral features

**Important**: This model deliberately **excludes** "states prefer local" to avoid circular logic. We want to identify what *else* predicts choice.

- Random Forest, Gradient Boosting, Logistic Regression, Decision Tree
- 5-fold cross-validation
- Feature importance analysis (4 methods)
- Consensus feature ranking
- Business interpretation

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl

# Run analyses (from repository root)
python scripts/pizza_analysis.py           # Descriptive stats
python scripts/competitive_model.py        # Competitive analysis
python scripts/regression_analysis.py      # Regression models
python scripts/ensemble_model.py           # ML model

# Generate visualizations
python scripts/generate_summary.py                 # fig1-6
python scripts/generate_competitive_visuals.py    # fig7-12
python scripts/generate_ml_visuals.py             # fig13-17
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

---

## Statistical Methods

| Method | Purpose | Reference |
|--------|---------|-----------|
| Chi-square | Preference distribution testing | Pearson (1900) |
| Friedman Test | Importance rating differences | Friedman (1937) |
| Multiple Regression | Loyalty predictors | OLS |
| Random Forest | Behavioral prediction | Breiman (2001) |
| Permutation Importance | Robust feature ranking | scikit-learn |

See `reports/METHODOLOGY.md` for full statistical justification.

---

## Business Recommendations

For a new local pizza entrant near RIT:

1. **Product**: Exceptional taste (non-negotiable), quality crust
2. **Price**: $17-20 for 16" pizza (match chains, signal quality)
3. **Service**: Fast pickup-focused model (<22 minutes)
4. **Position**: "Local quality at chain convenience"
5. **Target**: Frequent orderers with transportation (they choose local)

---

## Authors

BANA255 Analytics Team, Fall 2025

---

## License

Academic use only. Data collected under IRB-approved survey protocol.
