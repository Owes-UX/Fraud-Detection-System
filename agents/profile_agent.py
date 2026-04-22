"""
ProfileAgent: Builds rolling per-user behavioural baselines (no lookahead).
"""
import pandas as pd
import numpy as np

class ProfileAgent:
    def __init__(self):
        self._history = {}

    def update_and_score(self, tx: pd.Series) -> dict:
        sid = tx["SenderID"]
        hist = self._history.get(sid, [])
        profile = {}

        if len(hist) == 0:
            profile["is_new_sender"] = 1
            profile["amount_zscore"] = 0.0
            profile["time_since_last_tx"] = -1
            profile["tx_count_1h"] = 0
            profile["tx_count_24h"] = 0
            profile["new_recipient"] = 1
            profile["new_payment_method"] = 1
            profile["night_activity"] = 1 if tx.get("hour", 12) in range(0, 6) else 0
        else:
            amounts = np.array([h["Amount"] for h in hist])
            mu, sigma = amounts.mean(), amounts.std() + 1e-9
            profile["is_new_sender"] = 0
            profile["amount_zscore"] = float((tx["Amount"] - mu) / sigma)

            last_ts = hist[-1]["Timestamp"]
            cur_ts = tx["Timestamp"]
            delta = (cur_ts - last_ts).total_seconds() if pd.notna(last_ts) and pd.notna(cur_ts) else -1
            profile["time_since_last_tx"] = float(delta)

            if pd.notna(cur_ts):
                profile["tx_count_1h"]  = sum(1 for h in hist if pd.notna(h["Timestamp"]) and (cur_ts - h["Timestamp"]).total_seconds() <= 3600)
                profile["tx_count_24h"] = sum(1 for h in hist if pd.notna(h["Timestamp"]) and (cur_ts - h["Timestamp"]).total_seconds() <= 86400)
            else:
                profile["tx_count_1h"] = profile["tx_count_24h"] = 0

            past_recipients = {h["RecipientID"] for h in hist if pd.notna(h.get("RecipientID"))}
            past_methods    = {h["PaymentMethod"] for h in hist if pd.notna(h.get("PaymentMethod"))}
            profile["new_recipient"]      = 1 if tx.get("RecipientID") not in past_recipients else 0
            profile["new_payment_method"] = 1 if tx.get("PaymentMethod") not in past_methods else 0
            profile["night_activity"]     = 1 if tx.get("hour", 12) in range(0, 6) else 0

        self._history.setdefault(sid, []).append({
            "Amount": tx["Amount"],
            "Timestamp": tx.get("Timestamp"),
            "RecipientID": tx.get("RecipientID"),
            "PaymentMethod": tx.get("PaymentMethod"),
        })
        return profile
