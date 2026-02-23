from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import numpy as np

def evaluate_models(train_features):
    X = train_features.drop(columns=["member_id", "churn"])
    y = train_features["churn"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=124),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    }

    roc_curves = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        print(f"{name}: AUC-ROC = {scores.mean():.4f} (+/- {scores.std():.4f})")

        y_proba = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_curves[name] = (fpr, tpr, scores.mean())

    return roc_curves

def train_model(train_features):
    X = train_features.drop(columns=["member_id", "churn"])
    y = train_features["churn"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

def predict(model, feature_names, test_features):
    X_no_outreach = test_features[feature_names].copy()
    X_no_outreach["outreach"] = 0

    X_with_outreach = test_features[feature_names].copy()
    X_with_outreach["outreach"] = 1

    prob_no_outreach = model.predict_proba(X_no_outreach)[:, 1]
    prob_with_outreach = model.predict_proba(X_with_outreach)[:, 1]

    results = test_features[["member_id", "has_E11_9", "has_I10", "has_Z71_3"]].copy()
    results["churn_prob_no_outreach"] = prob_no_outreach
    results["churn_prob_with_outreach"] = prob_with_outreach
    results["outreach_benefit"] = prob_no_outreach - prob_with_outreach
    results["model_score"] = results["churn_prob_no_outreach"] * results["outreach_benefit"]
    results["icd_boost"] = (results["has_E11_9"] + results["has_I10"] + results["has_Z71_3"]) * 0.05
    results["prioritization_score"] = results["model_score"] + results["icd_boost"]
    results = results.sort_values("prioritization_score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    return results

def find_optimal_n(results):
    # select members whose churn probability without outreach exceeds 0.3
    # this threshold aligns with the observed churn rate of ~20% in training data
    at_risk = results[results["churn_prob_no_outreach"] > 0.3]
    optimal_n = len(at_risk)

    # sort benefits descending for plotting
    benefits = np.sort(results["outreach_benefit"].clip(lower=0).values)[::-1]

    # smooth using a rolling average to reduce noise
    window = 50
    smoothed_benefits = np.convolve(benefits, np.ones(window) / window, mode="valid")

    return optimal_n, benefits, smoothed_benefits
