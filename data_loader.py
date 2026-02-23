import pandas as pd
import os

def load_train_data():
    train_path = "train_data"

    app_usage = pd.read_csv(os.path.join(train_path, "app_usage.csv"))
    churn_labels = pd.read_csv(os.path.join(train_path, "churn_labels.csv"))
    claims = pd.read_csv(os.path.join(train_path, "claims.csv"))
    web_visits = pd.read_csv(os.path.join(train_path, "web_visits.csv"))

    return app_usage, churn_labels, claims, web_visits

def load_test_data():
    test_path = "test_data"

    app_usage = pd.read_csv(os.path.join(test_path, "test_app_usage.csv"))
    members = pd.read_csv(os.path.join(test_path, "test_members.csv"))
    claims = pd.read_csv(os.path.join(test_path, "test_claims.csv"))
    web_visits = pd.read_csv(os.path.join(test_path, "test_web_visits.csv"))

    return app_usage, members, claims, web_visits
