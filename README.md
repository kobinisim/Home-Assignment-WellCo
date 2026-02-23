# WellCo Churn Prevention — Prioritized Outreach Solution

## Overview
This project addresses WellCo's member churn problem by building a machine learning pipeline that produces a ranked list of members for prioritized outreach. The solution identifies members who are at high risk of churning and would benefit most from being contacted, with additional prioritization for clinically relevant members (ICD-10: E11.9, I10, Z71.3).

---

## Setup

### Requirements
Install the required Python libraries:
```
pip install pandas matplotlib scikit-learn xgboost
```

### Project Structure
```
pythonProject/
│
├── main.py                  # Entry point — runs the full pipeline
├── data_loader.py           # Loads train and test CSV files
├── feature_engineering.py   # Builds per-member feature tables
├── model.py                 # Model training, evaluation, prediction, optimal n
├── visualization.py         # All plots
│
├── train_data/              # Training CSV files
│   ├── app_usage.csv
│   ├── claims.csv
│   ├── web_visits.csv
│   └── churn_labels.csv
│
├── test_data/               # Test CSV files
│   ├── test_app_usage.csv
│   ├── test_claims.csv
│   ├── test_web_visits.csv
│   └── test_members.csv
│
├── train_visualization/     # Generated training plots
├── test_results/            # Generated output files
```

### How to Run
```
python main.py
```
This runs the full pipeline end-to-end and produces:
- Plots saved to `train_visualization/`
- `test_results/outreach_list.csv` — the ranked list of top n members
- `test_results/optimal_n.png` — the optimal n plot

---

## Approach

### 1. Feature Engineering
One row per member is built by combining all four data sources:

| Feature | Source | Rationale |
|---|---|---|
| `days_as_member` | churn_labels | Newer members churn more |
| `session_count` | app_usage | App engagement signals retention |
| `has_E11_9`, `has_I10`, `has_Z71_3` | claims | Priority clinical populations |
| `count_E11_9`, `count_I10`, `count_Z71_3` | claims | Frequency of priority diagnosis encounters |
| `num_of_total_claims` | claims | Overall healthcare engagement |
| `health_web_visits` | web_visits | Health content engagement reduces churn |
| `non_health_web_visits` | web_visits | Non-health browsing increases churn risk |
| `total_web_visits` | web_visits | Overall platform engagement |
| `outreach` | churn_labels | Outreach reduces churn — incorporated as a feature |

### 2. Outreach Data
Outreach occurred between the observation period and the churn measurement window. It is included as a feature so the model learns how much outreach reduces churn probability. At prediction time, each test member is scored twice — once with `outreach=0` and once with `outreach=1` — to estimate the individual benefit of outreach for each member.

### 3. Model Selection
Three models were evaluated using 5-fold cross-validation with AUC-ROC:

| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.6740 |
| Random Forest | 0.6335 |
| XGBoost | 0.6167 |

**Logistic Regression** was selected as the final model — it achieved the highest AUC-ROC and the lowest variance across folds, indicating stable and consistent performance.

### 4. Prioritization Score
Each test member receives a `prioritization_score` composed of:

- **`model_score`** = `churn_prob_no_outreach × outreach_benefit`
  Rewards members who are both high churn risk AND respond well to outreach
- **`icd_boost`** = `(has_E11_9 + has_I10 + has_Z71_3) × 0.05`
  A clinical priority bonus for members with WellCo's focus diagnoses

```
prioritization_score = model_score + icd_boost
```

### 5. Optimal n
The optimal outreach size is determined by selecting members whose predicted churn probability without outreach exceeds **0.3**. This threshold is grounded in the observed churn rate of ~20% in the training data, ensuring we target members who are genuinely at risk rather than the entire population.

Members are first filtered by this threshold, then ranked by `prioritization_score`. This two-step approach is important — without the threshold filter, members with low churn risk but high ICD boost could appear in the top n despite not being genuine churn risks. By filtering first, we guarantee every member in the output has a real predicted churn risk above 30%, and the ranking within that group is driven by both model score and clinical priority.

---

## Output
`test_results/outreach_list.csv` contains the following columns:

| Column | Description |
|---|---|
| `member_id` | Member identifier |
| `has_E11_9` | 1 if member has type 2 diabetes diagnosis |
| `has_I10` | 1 if member has hypertension diagnosis |
| `has_Z71_3` | 1 if member has dietary counseling diagnosis |
| `churn_prob_no_outreach` | Predicted churn probability without outreach |
| `churn_prob_with_outreach` | Predicted churn probability with outreach |
| `outreach_benefit` | Reduction in churn probability due to outreach |
| `model_score` | churn_prob_no_outreach × outreach_benefit |
| `icd_boost` | Clinical priority bonus |
| `prioritization_score` | Final ranking score |
| `rank` | Member rank (1 = highest priority) |
