import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from llm_client import LLMClient


DATA_DIR = "data"
MODEL_PATH = "models/fraud_model.pkl"
RANDOM_STATE = 42


def load_data():
    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    users_path = os.path.join(DATA_DIR, "users.json")
    locations_path = os.path.join(DATA_DIR, "locations.json")
    sms_path = os.path.join(DATA_DIR, "sms.json")
    mails_path = os.path.join(DATA_DIR, "mails.json")

    tx = pd.read_csv(tx_path)
    users = pd.read_json(users_path)
    locations = pd.read_json(locations_path)
    sms = pd.read_json(sms_path)
    mails = pd.read_json(mails_path)

    return tx, users, locations, sms, mails


def build_user_features(
    users: pd.DataFrame,
    sms: pd.DataFrame,
    mails: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per account using IBAN as key.
    users.json columns: ['first_name','last_name','birth_year','salary','job','iban','residence','description']
    """

    if "iban" not in users.columns:
        raise ValueError("Expected 'iban' column in users.json")

    user_feat = users.copy().set_index("iban")

    # --- SMS suspicious count, grouped by iban if present ---
    if "iban" in sms.columns and "text" in sms.columns:
        sms["is_suspicious"] = sms["text"].astype(str).str.contains(
            "password|verify|urgent|account locked|otp",
            case=False,
            na=False,
        )
        sms_feat = sms.groupby("iban")["is_suspicious"].sum().rename(
            "num_suspicious_sms"
        )
        user_feat = user_feat.join(sms_feat, how="left")
    else:
        user_feat["num_suspicious_sms"] = 0

    # --- Email suspicious count, grouped by iban if present ---
    if "iban" in mails.columns and "subject" in mails.columns:
        mails["is_suspicious"] = mails["subject"].astype(str).str.contains(
            "password|verify|urgent|account locked|otp",
            case=False,
            na=False,
        )
        mail_feat = mails.groupby("iban")["is_suspicious"].sum().rename(
            "num_suspicious_mails"
        )
        user_feat = user_feat.join(mail_feat, how="left")
    else:
        user_feat["num_suspicious_mails"] = 0

    # --- Location diversity, grouped by iban if present ---
    if "iban" in locations.columns and "country" in locations.columns:
        loc_countries = (
            locations.groupby("iban")["country"]
            .nunique()
            .rename("num_countries_seen")
        )
        user_feat = user_feat.join(loc_countries, how="left")
    else:
        user_feat["num_countries_seen"] = 0

    user_feat = user_feat.fillna(0).reset_index().rename(columns={"iban": "account_key"})
    return user_feat


def add_user_features_to_transactions(
    tx: pd.DataFrame, user_feat: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge sender user features into transactions using sender_iban.
    """

    if "sender_iban" not in tx.columns:
        raise ValueError("Expected 'sender_iban' column in transactions.csv")

    if "account_key" not in user_feat.columns:
        raise ValueError("Expected 'account_key' column in user_feat")

    merged = tx.merge(
        user_feat.add_prefix("sender_"),
        left_on="sender_iban",
        right_on="sender_account_key",
        how="left",
    )

    merged = merged.fillna(0)
    return merged


def add_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a weakly supervised label 'is_fraud' using simple rules.
    """
    df["is_fraud"] = 0

    # Rule 1: very high amount
    if "amount" in df.columns:
        df.loc[df["amount"] > 5000, "is_fraud"] = 1

    # Rule 2: suspicious transaction types
    suspicious_types = ["chargeback", "dispute", "reversal"]
    if "transaction_type" in df.columns:
        df.loc[df["transaction_type"].isin(suspicious_types), "is_fraud"] = 1

    # Rule 3: many suspicious sms/emails
    if "sender_num_suspicious_sms" in df.columns:
        df.loc[df["sender_num_suspicious_sms"] > 3, "is_fraud"] = 1
    if "sender_num_suspicious_mails" in df.columns:
        df.loc[df["sender_num_suspicious_mails"] > 3, "is_fraud"] = 1

    return df


def add_llm_scores(df: pd.DataFrame, llm: LLMClient) -> pd.DataFrame:
    """
    Use LLM to score transaction description text for fraud risk.
    """
    risks: List[float] = []
    descriptions = df["description"].fillna("").astype(str)

    for text in descriptions:
        if not text.strip():
            risks.append(0.0)
        else:
            risks.append(llm.score_text_risk(text))

    df["llm_text_risk"] = risks
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Select numeric features for training.
    """
    feature_cols = []

    for col in [
        "amount",
        "sender_num_suspicious_sms",
        "sender_num_suspicious_mails",
        "sender_num_countries_seen",
        "llm_text_risk",
    ]:
        if col in df.columns:
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No feature columns found")

    X = df[feature_cols].astype(float)
    y = df["is_fraud"].astype(int)

    return X, y, feature_cols


def train_and_save_model(X, y, feature_cols: List[str]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"model": clf, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")


def main():
    print("Loading data...")
    tx, users, locations, sms, mails = load_data()

    print("Building user features...")
    user_feat = build_user_features(users, sms, mails, locations)

    print("Merging user features into transactions...")
    df = add_user_features_to_transactions(tx, user_feat)

    print("Adding weak labels...")
    df = add_weak_labels(df)

    print("Adding LLM text risk scores...")
    llm = LLMClient()
    df = add_llm_scores(df, llm)

    print("Building feature matrix...")
    X, y, feature_cols = build_feature_matrix(df)

    print("Training model...")
    train_and_save_model(X, y, feature_cols)

    print("Done.")


if __name__ == "__main__":
    main()