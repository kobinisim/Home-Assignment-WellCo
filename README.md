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

## Discussion

### Feature Selection
Features were selected based on three criteria: **domain relevance**, **data quality**, and **predictive power**.

**Domain relevance:** WellCo's mission is preventive health engagement. Features were chosen to capture two dimensions — clinical engagement (ICD codes, claims frequency) and digital engagement (app sessions, web visits). Members who are actively engaged with health content and clinical care are more likely to stay. We separated web visits into health-related and non-health-related categories based on the content taxonomy in the client brief, as non-health browsing signals disengagement from WellCo's core purpose.

**Data quality:** All features are counts or binary flags derived directly from structured data with no missing values after aggregation. Members with no activity in a given source were assigned zero, which is a meaningful and accurate value.

**Predictive power:** After training, the model coefficients confirmed the expected directions — outreach, health engagement, and clinical encounters all reduce churn, while non-health browsing increases it. `days_as_member` emerged as a strong signal: newer members churn more, consistent with typical subscription retention patterns.

Priority ICD codes (E11.9, I10, Z71.3) were included both as binary flags (has the diagnosis) and counts (how many times the code appears), since frequency of encounters carries additional signal beyond simple presence.

---

### Model Evaluation
Model performance was evaluated using **5-fold cross-validation** with **AUC-ROC** as the primary metric.

**Why AUC-ROC:** Our goal is not to classify members as churned or not, but to **rank** them by churn risk. AUC-ROC directly measures the model's ability to rank a churned member above a non-churned member, making it the most appropriate metric for this task. It is also threshold-independent, which suits our use case since we apply our own threshold (churn probability > 0.3) separately from the model itself.

**Why cross-validation:** Using cross-validation on the training data provides a reliable estimate of real-world performance without touching the test data. It also reveals model stability — a low standard deviation across folds (as seen with Logistic Regression at ±0.007) indicates consistent performance.

Three models were compared — Logistic Regression, Random Forest, and XGBoost. Logistic Regression outperformed the more complex models, suggesting the relationship between features and churn is predominantly linear. It was selected as the final model.

---

### Using Outreach Data in Modelling
The outreach event occurred **between** the observation period and the churn measurement window, meaning it could have directly influenced whether a member churned. Including it as a feature allows the model to learn the effect of outreach on churn probability.

At prediction time, every test member is scored **twice** — once with `outreach=0` (no contact) and once with `outreach=1` (contacted). The difference between the two predicted churn probabilities is the `outreach_benefit` — how much contacting this specific member would reduce their churn risk.

An important finding: analysis of the training data showed that **outreach was applied randomly**, not targeted at high-risk or priority-diagnosis members. This means the model learned the average effect of outreach across all member types, not a targeted effect. As a result, outreach benefit is relatively uniform across members. In a real deployment, targeted outreach would likely produce stronger effects than what the model learned from random outreach.

---

### Selecting n (Outreach Size)
The optimal n is **not driven by cost alone**. Three factors were considered:

1. **Cost:** Outreach has a constant marginal cost per member. We should only contact members where the expected benefit justifies this cost — members who are genuinely at risk of churning and would respond to outreach.

2. **Churn risk threshold:** The 0.3 threshold was chosen pragmatically — it produces a list of approximately 20% of test members, consistent with the observed churn rate in training data. It is a business decision balancing outreach cost against coverage. A lower threshold would reach more members at higher cost; a higher threshold would be more selective but risk missing genuine churners. This threshold should be revisited and calibrated as more data becomes available.

3. **Two-step filtering:** Members are first filtered by the churn probability threshold, then ranked by `prioritization_score`. This prevents members with low churn risk but high clinical priority (ICD boost) from appearing in the outreach list — clinical priority should influence ranking among at-risk members, not override the risk requirement itself.

This approach yields **n = 1,996 members** — approximately 20% of the test population, consistent with the observed churn rate.

---
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
