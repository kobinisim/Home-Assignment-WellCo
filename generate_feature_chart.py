import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data_loader import load_train_data
from feature_engineering import build_features
from sklearn.linear_model import LogisticRegression

app_usage, churn_labels, claims, web_visits = load_train_data()
train_features = build_features(app_usage, churn_labels, claims, web_visits)

X = train_features.drop(columns=["member_id", "churn"])
y = train_features["churn"]

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

feature_names = X.columns.tolist()
coefficients = list(zip(feature_names, model.coef_[0]))
coefficients = sorted(coefficients, key=lambda x: x[1])

labels = [item[0] for item in coefficients]
values = [item[1] for item in coefficients]
colors = ["steelblue" if v < 0 else "salmon" for v in values]

plt.figure(figsize=(10, 8))
bars = plt.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)

for bar, value in zip(bars, values):
    plt.text(value + (0.002 if value >= 0 else -0.002),
             bar.get_y() + bar.get_height() / 2,
             f"{value:.4f}",
             va="center",
             ha="left" if value >= 0 else "right",
             fontsize=8)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient Value")
plt.title("Logistic Regression Coefficients\n(negative = reduces churn, positive = increases churn)")
plt.legend(handles=[
    mpatches.Patch(color="steelblue", label="Reduces churn risk"),
    mpatches.Patch(color="salmon", label="Increases churn risk")
])
plt.tight_layout()
plt.savefig("train_visualization/logistic_regression_coefficients.png", dpi=150)
plt.close()
print("saved: train_visualization/logistic_regression_coefficients.png")
