"""
ReplyMirror Fraud Detection – Owes + LLM agents version.

Usage:
    python main.py --data ./data --output ./output/predictions.txt
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.profile_agent import ProfileAgent
from agents.risk_agent import RiskAgent
from agents.decision_agent import DecisionAgent
from agents.text_agent import TextAgent
from utils.io import write_output


def score_dataframe(
    transactions: pd.DataFrame,
    feature_agent: FeatureAgent,
    profile_agent: ProfileAgent,
    risk_agent: RiskAgent,
):
    all_ids, all_scores = [], []
    for i, (_, tx) in enumerate(transactions.iterrows()):
        feats = feature_agent.extract(tx)
        profile = profile_agent.update_and_score(
            pd.Series(
                {
                    "SenderID": tx["sender_id"],
                    "RecipientID": tx["recipient_id"],
                    "Timestamp": tx["timestamp"],
                    "Amount": tx["amount"],
                    "PaymentMethod": tx["payment_method"],
                }
            )
        )
        record = {**feats, **profile}
        s = risk_agent.score(record)
        all_ids.append(tx["transaction_id"])
        all_scores.append(s)
        if (i + 1) % 20 == 0:
            print(f"  Scored {i+1}/{len(transactions)}...")
    return all_ids, all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="./output/predictions.txt")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--flag-rate", type=float, default=0.15)  # aim ~15%
    args = parser.parse_args()

    # Ensure output folder exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1) Load data
    data_agent = DataAgent(args.data)
    data_agent.load()
    tx = data_agent.transactions
    print(f"[Main] Loaded {len(tx)} transactions")

    # 2) Build LLM text agent (uses SMS + mails)
    text_agent = TextAgent(sms=data_agent.sms, mails=data_agent.mails)

    # 3) Build feature + profile + risk agents
    feature_agent = FeatureAgent(
        users=data_agent.users,
        locations=data_agent.locations,
        sms=data_agent.sms,
        mails=data_agent.mails,
        text_agent=text_agent,
    )

    risk_agent = RiskAgent()
    profile_for_train = ProfileAgent()

    # 4) Train anomaly model on whole dataset (unsupervised)
    print("[Main] Building training records...")
    train_records = []
    for _, row in tx.iterrows():
        feats = feature_agent.extract(row)
        profile = profile_for_train.update_and_score(
            pd.Series(
                {
                    "SenderID": row["sender_id"],
                    "RecipientID": row["recipient_id"],
                    "Timestamp": row["timestamp"],
                    "Amount": row["amount"],
                    "PaymentMethod": row["payment_method"],
                }
            )
        )
        train_records.append({**feats, **profile})

    print(f"[Main] Training anomaly model on {len(train_records)} records...")
    risk_agent.train_anomaly(train_records)

    # 5) Score all transactions
    print(f"[Main] Scoring {len(tx)} transactions...")
    profile_eval = ProfileAgent()
    all_ids, all_scores = score_dataframe(tx, feature_agent, profile_eval, risk_agent)
    if all_scores:
        print(f"[Main] Example score: {all_ids[0]} -> {all_scores[0]:.4f}")

    # 6) Decide which transactions to flag
    decision_agent = DecisionAgent(
        base_threshold=args.threshold, target_flag_rate=args.flag_rate
    )
    flagged = decision_agent.decide(all_ids, all_scores)
    print(f"[Main] Flagged {len(flagged)} transactions")

    # Safety: if still zero, take top 5 highest scores
    if len(flagged) == 0 and len(all_ids) > 0:
        print("[Main] No flags under threshold; forcing top 5 by score.")
        sorted_pairs = sorted(
            zip(all_ids, all_scores), key=lambda x: x[1], reverse=True
        )
        flagged = [tid for tid, _ in sorted_pairs[:5]]

    # 7) Write output file
    print(f"[Main] Writing output to {args.output} ...")
    write_output(flagged, args.output)
    print(f"[Main] Done. Wrote {len(flagged)} IDs.")


if __name__ == "__main__":
    main()