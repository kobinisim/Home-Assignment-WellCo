from data_loader import load_train_data, load_test_data
from visualization import plot_sessions_per_member, plot_claims_pie, plot_web_visits_bar, plot_churn_vs_outreach, plot_outreach_by_diagnosis, plot_optimal_n, plot_roc_curves
from feature_engineering import build_features, build_test_features
from model import train_model, predict, find_optimal_n, evaluate_models
import os

app_usage, churn_labels, claims, web_visits = load_train_data()

plot_sessions_per_member(app_usage)
plot_claims_pie(claims)
plot_web_visits_bar(web_visits)
plot_churn_vs_outreach(churn_labels)

train_features = build_features(app_usage, churn_labels, claims, web_visits)
plot_outreach_by_diagnosis(train_features)


roc_curves = evaluate_models(train_features)
plot_roc_curves(roc_curves)

model, feature_names = train_model(train_features)
print("model trained successfully")

coefficients = zip(feature_names, model.coef_[0])
for feature, coef in sorted(coefficients, key=lambda x: abs(x[1]), reverse=True):
    print(f"{feature}: {coef:.4f}")

test_app_usage, test_members, test_claims, test_web_visits = load_test_data()
test_features = build_test_features(test_app_usage, test_members, test_claims, test_web_visits)

results = predict(model, feature_names, test_features)

optimal_n, benefits, smoothed_benefits = find_optimal_n(results)
print(f"optimal n: {optimal_n}")
plot_optimal_n(benefits, smoothed_benefits, optimal_n)


top_n = results[results["churn_prob_no_outreach"] > 0.3].copy()
top_n["rank"] = range(1, len(top_n) + 1)
os.makedirs("test_results", exist_ok=True)
top_n.to_csv("test_results/outreach_list.csv", index=False)
print(f"saved top {len(top_n)} members to test_results/outreach_list.csv")


