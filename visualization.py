import matplotlib.pyplot as plt
import os

def plot_sessions_per_member(app_usage):
    sessions_per_member = app_usage.groupby("member_id").size()

    mean = sessions_per_member.mean()
    std = sessions_per_member.std()

    plt.figure(figsize=(10, 5))
    plt.hist(sessions_per_member, bins=30, edgecolor="black")
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.1f}")
    plt.axvline(mean + std, color="orange", linestyle="--", label=f"Mean + STD: {mean + std:.1f}")
    plt.axvline(mean - std, color="orange", linestyle="--", label=f"Mean - STD: {mean - std:.1f}")
    plt.title("Distribution of Sessions per Member")
    plt.xlabel("Number of Sessions")
    plt.ylabel("Number of Members")
    plt.legend()
    plt.tight_layout()

    os.makedirs("train_visualization", exist_ok=True)
    plt.savefig("train_visualization/app_usage_hist_train data.png")

    plt.close()

def plot_claims_pie(claims):
    icd_counts = claims["icd_code"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(icd_counts.values, labels=icd_counts.index, autopct="%1.1f%%")
    plt.title("Distribution of Diagnoses (ICD Codes)")
    plt.tight_layout()

    os.makedirs("train_visualization", exist_ok=True)
    plt.savefig("train_visualization/claims_pie_train_data.png")

    plt.close()

def plot_web_visits_bar(web_visits):
    health_titles = {
        "Diabetes management", "Hypertension basics", "Stress reduction",
        "Restorative sleep tips", "Healthy eating guide", "Aerobic exercise",
        "HbA1c targets", "Strength training basics", "Lowering blood pressure",
        "Sleep hygiene", "Cardio workouts", "Mediterranean diet",
        "Exercise routines", "Meditation guide", "Cardiometabolic health",
        "High-fiber meals", "Cholesterol friendly foods", "Weight management"
    }

    title_counts = web_visits["title"].value_counts()
    colors = ["steelblue" if title in health_titles else "salmon" for title in title_counts.index]

    plt.figure(figsize=(10, 10))
    plt.barh(title_counts.index, title_counts.values, color=colors)
    plt.xlabel("Number of Visits")
    plt.title("Web Visits by Title")
    plt.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="Health"),
        plt.Rectangle((0, 0), 1, 1, color="salmon", label="Non-Health")
    ])
    plt.tight_layout()

    os.makedirs("train_visualization", exist_ok=True)
    plt.savefig("train_visualization/web_visits_bar_train_data.png")

    plt.close()

def plot_churn_vs_outreach(churn_labels):
    churned = churn_labels[churn_labels["churn"] == 1]
    not_churned = churn_labels[churn_labels["churn"] == 0]

    churned_outreach = [
        (churned["outreach"] == 0).sum(),
        (churned["outreach"] == 1).sum()
    ]
    not_churned_outreach = [
        (not_churned["outreach"] == 0).sum(),
        (not_churned["outreach"] == 1).sum()
    ]

    x = [0, 1]
    width = 0.35

    plt.figure(figsize=(8, 6))
    bars1 = plt.bar([i - width/2 for i in x], churned_outreach, width=width, color="salmon", label="Churned")
    bars2 = plt.bar([i + width/2 for i in x], not_churned_outreach, width=width, color="steelblue", label="Not Churned")
    plt.xticks(x, ["No Outreach", "Outreach"])

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(int(bar.get_height())), ha="center")
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(int(bar.get_height())), ha="center")
    plt.ylabel("Number of Members")
    plt.title("Churn vs Outreach")
    plt.legend()
    plt.tight_layout()

    os.makedirs("train_visualization", exist_ok=True)
    plt.savefig("train_visualization/churn_vs_outreach_train_data.png")

    plt.close()

def plot_outreach_by_diagnosis(train_features):
    diagnoses = {"E11.9": "has_E11_9", "I10": "has_I10", "Z71.3": "has_Z71_3"}

    outreach_rates_with = []
    outreach_rates_without = []

    for label, col in diagnoses.items():
        with_diag = train_features[train_features[col] == 1]
        without_diag = train_features[train_features[col] == 0]
        outreach_rates_with.append(with_diag["outreach"].mean() * 100)
        outreach_rates_without.append(without_diag["outreach"].mean() * 100)

    x = [0, 1, 2]
    width = 0.35

    plt.figure(figsize=(8, 6))
    bars1 = plt.bar([i - width/2 for i in x], outreach_rates_with, width=width, color="steelblue", label="Has Diagnosis")
    bars2 = plt.bar([i + width/2 for i in x], outreach_rates_without, width=width, color="salmon", label="No Diagnosis")

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center")
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center")

    plt.xticks(x, list(diagnoses.keys()))
    plt.ylabel("Outreach Rate (%)")
    plt.title("Outreach Rate by Priority Diagnosis")
    plt.legend()
    plt.tight_layout()

    os.makedirs("train_visualization", exist_ok=True)
    plt.savefig("train_visualization/outreach_by_diagnosis_train_data.png")

    plt.close()

def plot_roc_curves(roc_curves):
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_score) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()

    os.makedirs("train_visualization", exist_ok=True)
    plt.savefig("train_visualization/roc_curves.png")

    plt.close()

def plot_optimal_n(benefits, smoothed_benefits, optimal_n):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(benefits) + 1), benefits, color="lightsteelblue", alpha=0.5, label="Outreach benefit per member (sorted descending)")
    plt.plot(range(1, len(smoothed_benefits) + 1), smoothed_benefits, color="steelblue", label="Smoothed benefit (rolling average)")
    plt.axvline(optimal_n, color="red", linestyle="--", label=f"Optimal n = {optimal_n}")
    plt.title("Outreach Benefit per Member — Sorted Descending")
    plt.xlabel("Number of Members (n)")
    plt.ylabel("Outreach Benefit (churn reduction)")
    plt.legend()
    plt.tight_layout()

    os.makedirs("test_results", exist_ok=True)
    plt.savefig("test_results/optimal_n.png")

    plt.close()
