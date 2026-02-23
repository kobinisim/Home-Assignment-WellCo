import pandas as pd

HEALTH_TITLES = {
    "Diabetes management", "Hypertension basics", "Stress reduction",
    "Restorative sleep tips", "Healthy eating guide", "Aerobic exercise",
    "HbA1c targets", "Strength training basics", "Lowering blood pressure",
    "Sleep hygiene", "Cardio workouts", "Mediterranean diet",
    "Exercise routines", "Meditation guide", "Cardiometabolic health",
    "High-fiber meals", "Cholesterol friendly foods", "Weight management"
}

def build_features(app_usage, churn_labels, claims, web_visits):
    features = churn_labels[["member_id", "signup_date", "churn", "outreach"]].copy()

    # convert signup_date to datetime and calculate how many days each member has been with WellCo
    features["signup_date"] = pd.to_datetime(features["signup_date"])
    reference_date = pd.Timestamp("2025-07-01")
    features["days_as_member"] = (reference_date - features["signup_date"]).dt.days
    features = features.drop(columns=["signup_date"])

    # app_usage: count sessions per member
    session_counts = app_usage.groupby("member_id").size().reset_index(name="session_count")
    features = features.merge(session_counts, on="member_id", how="left")
    features["session_count"] = features["session_count"].fillna(0).astype(int)

    # claims: for each member, check if they appear anywhere in claims with icd_code E11.9, I10 and Z71.3. Mark them 1 if yes, 0 if no
    features["has_E11_9"] = features["member_id"].isin(claims[claims["icd_code"] == "E11.9"]["member_id"]).astype(int)
    features["has_I10"] = features["member_id"].isin(claims[claims["icd_code"] == "I10"]["member_id"]).astype(int)
    features["has_Z71_3"] = features["member_id"].isin(claims[claims["icd_code"] == "Z71.3"]["member_id"]).astype(int)

    # claims: count how many times each priority icd code appears per member
    for code, col_name in [("E11.9", "count_E11_9"), ("I10", "count_I10"), ("Z71.3", "count_Z71_3")]:
        code_counts = claims[claims["icd_code"] == code].groupby("member_id").size().reset_index(name=col_name)
        features = features.merge(code_counts, on="member_id", how="left")
        features[col_name] = features[col_name].fillna(0).astype(int)

    num_claims = claims.groupby("member_id").size().reset_index(name="num_of_total_claims")
    features = features.merge(num_claims, on="member_id", how="left")
    features["num_of_total_claims"] = features["num_of_total_claims"].fillna(0).astype(int)

    # web_visits: count total, health, and non-health visits per member
    total_visits = web_visits.groupby("member_id").size().reset_index(name="total_web_visits")
    health_visits = web_visits[web_visits["title"].isin(HEALTH_TITLES)].groupby("member_id").size().reset_index(name="health_web_visits")
    features = features.merge(total_visits, on="member_id", how="left")
    features = features.merge(health_visits, on="member_id", how="left")
    features["total_web_visits"] = features["total_web_visits"].fillna(0).astype(int)
    features["health_web_visits"] = features["health_web_visits"].fillna(0).astype(int)
    features["non_health_web_visits"] = features["total_web_visits"] - features["health_web_visits"]

    return features

def build_test_features(app_usage, members, claims, web_visits):
    features = members[["member_id", "signup_date"]].copy()

    # convert signup_date to datetime and calculate how many days each member has been with WellCo
    features["signup_date"] = pd.to_datetime(features["signup_date"])
    reference_date = pd.Timestamp("2025-07-01")
    features["days_as_member"] = (reference_date - features["signup_date"]).dt.days
    features = features.drop(columns=["signup_date"])

    # outreach is set to 0 for all test members since no outreach has happened yet
    features["outreach"] = 0

    # app_usage: count sessions per member
    session_counts = app_usage.groupby("member_id").size().reset_index(name="session_count")
    features = features.merge(session_counts, on="member_id", how="left")
    features["session_count"] = features["session_count"].fillna(0).astype(int)

    # claims: for each member, check if they appear anywhere in claims with icd_code E11.9, I10 and Z71.3. Mark them 1 if yes, 0 if no
    features["has_E11_9"] = features["member_id"].isin(claims[claims["icd_code"] == "E11.9"]["member_id"]).astype(int)
    features["has_I10"] = features["member_id"].isin(claims[claims["icd_code"] == "I10"]["member_id"]).astype(int)
    features["has_Z71_3"] = features["member_id"].isin(claims[claims["icd_code"] == "Z71.3"]["member_id"]).astype(int)

    # claims: count how many times each priority icd code appears per member
    for code, col_name in [("E11.9", "count_E11_9"), ("I10", "count_I10"), ("Z71.3", "count_Z71_3")]:
        code_counts = claims[claims["icd_code"] == code].groupby("member_id").size().reset_index(name=col_name)
        features = features.merge(code_counts, on="member_id", how="left")
        features[col_name] = features[col_name].fillna(0).astype(int)

    num_claims = claims.groupby("member_id").size().reset_index(name="num_of_total_claims")
    features = features.merge(num_claims, on="member_id", how="left")
    features["num_of_total_claims"] = features["num_of_total_claims"].fillna(0).astype(int)

    # web_visits: count total, health, and non-health visits per member
    total_visits = web_visits.groupby("member_id").size().reset_index(name="total_web_visits")
    health_visits = web_visits[web_visits["title"].isin(HEALTH_TITLES)].groupby("member_id").size().reset_index(name="health_web_visits")
    features = features.merge(total_visits, on="member_id", how="left")
    features = features.merge(health_visits, on="member_id", how="left")
    features["total_web_visits"] = features["total_web_visits"].fillna(0).astype(int)
    features["health_web_visits"] = features["health_web_visits"].fillna(0).astype(int)
    features["non_health_web_visits"] = features["total_web_visits"] - features["health_web_visits"]

    return features
