# 🛢️ Vice City Variables 2026
Predicting Heavy-Oil Well Failures Using Survival Analysis
Chirality x Canary Labs x DSMLC Datathon 2026

Transforming five years of oilfield time-series data into actionable operational intelligence for reliability optimization, predictive maintenance, and risk-based decision making.

Team Members
- Gurm
- Tom
- Toheeb
- KTGster
Vice City Variables Team

# 🎯 Problem Statement

The oil and gas industry generates massive volumes of production, maintenance, and operational data every day. Despite this wealth of information, maintenance activities are often reactive, resulting in:

- Unplanned production downtime
- Lost revenue
- Emergency repair costs
- Reduced equipment reliability
- Inefficient allocation of maintenance resources

As data consultants working with Strathcona Resources, our objective is to identify opportunities for operational optimization using five years of production and maintenance data.

# ❓ Research Question
- Can survival analysis predict heavy-oil well failures before they occur and identify recurring reliability risks, enabling operators to prioritize wells based on failure probability, expected downtime, and operational impact?

# 📌 Project Objectives

Our solution aims to:

- ✅ Estimate the probability of future well failures

- ✅ Predict time-to-failure after maintenance activities

- ✅ Identify wells exhibiting recurring reliability issues

- ✅ Quantify operational factors contributing to failures

- ✅ Support proactive maintenance planning

- ✅ Reduce downtime and maintenance costs

# 🔍 Dataset Overview

Provided by Strathcona Resources through the Chirality Research Datathon.

Available Data Sources
- Daily Production Data by Well
- Production Failure Records
- Maintenance Activities
- Equipment Repair History
- Operational Performance Metrics
- Time Horizon
- 5 Years of Historical Data
- Data Characteristics
- Daily Time-Series Observations
- Thousands of Repair Events
- Active and Failed Well Runs
- Operational and Geographic Attributes

# 🧠 Why Survival Analysis?

Traditional machine learning approaches struggle with right-censored data.

Many wells were still operating at the end of the observation period.

If active wells are:

- Removed → predictions become biased toward failures.
- Treated as failures → healthy wells are misrepresented.

Survival analysis naturally accounts for censored observations and provides:

Probability of failure over time
Remaining useful life estimates
Hazard/risk trends
Explainable maintenance prioritization

# ⚙️ Methodology
- 1. Data Engineering
Failure Proxy

Because formal failure logging only began recently, repair jobs were used as a failure proxy.

Validation showed strong overlap between:

Logged failures
Repair events

making repairs a reasonable reliability indicator.

- 2. Survival Modelling
Statistical Models
Kaplan-Meier Survival Estimator
Weibull Survival Analysis
Cox Proportional Hazards Model
Machine Learning Models
Boosted Cox Model
Survival Forest Approaches
Risk Ranking Models

- 3. Explainability

To ensure recommendations are actionable for engineers:

SHAP Values
Feature Importance Analysis
Risk Factor Attribution were incorporated into the pipeline.

# 📈 Key Findings
- Finding 1: Failures Are Highly Concentrated

A small subset of wells drives a disproportionately large share of future maintenance risk.

This creates an opportunity for targeted intervention.

Finding 2: Highest Risk Occurs After Repairs

Survival curves indicate that wells are most vulnerable immediately following a repair event.

This suggests:

Repair quality concerns
Unresolved root causes
Restart stress
Component degradation
Finding 3: Operational Factors Matter

Increased failure likelihood is associated with:

Higher production rates
Increased pump intensity
Specific repair types
Geographic operating conditions
Finding 4: Rod-Related Repairs Are Critical

Research and domain literature indicate rod failures represent one of the leading causes of workovers in sucker-rod pumping systems.

Wells experiencing rod-related repairs often fail again sooner than wells repaired for other components.

# 📊 Model Results
Metric	ResultMedian Well Life After Repair	67 Days
Best Model	Boosted Cox Model
Concordance Index (C-Index)	0.677
Explainability	SHAP

The Boosted Cox model delivered the strongest predictive performance while maintaining interpretability for field engineers and maintenance teams.

# 🏗️ Solution Architecture
                    ┌─────────────────────┐
                    │ Strathcona Datasets │
                    └──────────┬──────────┘
                               │
                               ▼
                ┌──────────────────────────┐
                │ Data Cleaning & ETL      │
                │ Pandas + PyArrow         │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Feature Engineering      │
                │ Failure / Run History    │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Survival Analysis Models │
                │ KM | Weibull | Cox       │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Risk Prediction Engine   │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ SHAP Explainability      │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Dashboard / Application  │
                └──────────────────────────┘

# 📂 Repository Structure
```
Vice-City-Variables-2026/
│
├── app/
│   ├── dashboard/
│   ├── visualizations/
│   └── application_code/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── survival_modeling.ipynb
│   ├── feature_engineering.ipynb
│   └── validation.ipynb
│
├── README.md
├── LICENSE
└── .gitignore
```
# 🚀 Future Development
Real-Time Condition Monitoring

Survival analysis provides strategic forecasting but does not capture rapidly developing equipment issues.

# Future enhancements include:

High-frequency sensor data integration
Vibration monitoring
Torque analysis
Motor current monitoring
Fast Fourier Transform (FFT) fault diagnostics
Dynamic Risk Scores

Combining:

Historical maintenance records
Survival probabilities
Real-time sensor telemetry

would enable continuously updated well reliability scores.

# 📚 References
Industry Research

Vishnuvardhan, S., Murthy, A. R., & Choudhary, A. (2023).

A Review on Pipeline Failures, Defects in Pipelines and Their Assessment and Fatigue Life Prediction Methods.

https://doi.org/10.1016/j.ijpvp.2022.104853

Equipment Reliability Research

Analysis of Workover Causes for Wells with Installed Downhole Sucker-Rod Pumps

https://stumejournals.com/journals/mtm/2019/2/66.full.pdf

🎥 Deliverables
✅ Public GitHub Repository
✅ Source Code
✅ Analysis Notebooks
✅ Presentation Slides
✅ 5-Minute Project Summary Video
✅ Research Methodology
✅ Validation and Findings
✅ AI Usage Disclosure

# 🏆 Datathon Competition

This project was developed for the:

Chirality Research × Canary Labs × DSMLC Datathon 2026

Theme: Operational Optimization Through Industrial Time-Series Analytics

Our goal is to transform raw operational data into actionable intelligence that helps operators move from reactive maintenance toward predictive reliability and data-driven decision making.

# ⭐ Key Takeaway

Not all wells fail equally. By combining survival analysis, explainable machine learning, and operational domain knowledge, we can identify high-risk assets before failure occurs and prioritize maintenance where it generates the greatest business value.
