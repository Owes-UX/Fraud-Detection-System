"""
RiskAgent: Fuses signals into a fraud score using anomaly detection,
supervised learning (if labels available), and heuristic rules.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

NUMERIC_COLS = [
    "amount", "log_amount", "balance", "balance_negative",
    "hour", "weekday", "tx_type_id", "pay_method_id",
    "iban_country_mismatch", "text_risk_score", "location_risk",
    "is_new_sender", "amount_zscore", "time_since_last_tx",
    "tx_count_1h", "tx_count_24h", "new_recipient",
    "new_payment_method", "night_activity",
]

class RiskAgent:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.anomaly_model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
        self.supervised_model = None
        self.scaler = StandardScaler()
        self._trained_anomaly = False
        self._trained_supervised = False

    def _to_array(self, records):
        return np.array([[float(r.get(c, 0) or 0) for c in NUMERIC_COLS] for r in records], dtype=float)

    def train_anomaly(self, records):
        if len(records) < 10: return
        X = self.scaler.fit_transform(self._to_array(records))
        self.anomaly_model.fit(X)
        self._trained_anomaly = True
        print(f"[RiskAgent] Anomaly model trained on {len(records)} samples.")

    def train_supervised(self, records, labels):
        import numpy as np
        y = np.array(labels)
        if len(records) < 20 or y.sum() < 5: return
        self.supervised_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42))
        ])
        self.supervised_model.fit(self._to_array(records), y)
        self._trained_supervised = True
        print(f"[RiskAgent] Supervised model trained. Fraud rate: {y.mean():.2%}")

    def score(self, record):
        x = np.array([[float(record.get(c, 0) or 0) for c in NUMERIC_COLS]])
        scores = []
        if self._trained_anomaly:
            raw = self.anomaly_model.score_samples(self.scaler.transform(x))[0]
            scores.append(float(np.clip((-raw - 0.2) / 0.5, 0, 1)) * 0.35)
        if self._trained_supervised:
            scores.append(float(self.supervised_model.predict_proba(x)[0][1]) * 0.50)
        h = self._heuristic(record)
        w = 0.30 if scores else 1.0
        scores.append(h * w)
        total_w = (0.35 if self._trained_anomaly else 0) + (0.50 if self._trained_supervised else 0) + w
        return float(np.clip(sum(scores) / total_w, 0, 1))

    def _heuristic(self, r):
        s = 0.0
        s += min(abs(float(r.get("amount_zscore", 0))) / 10.0, 0.4)
        s += min(float(r.get("tx_count_1h", 0)) / 10.0, 0.2)
        if r.get("new_recipient") and r.get("night_activity"): s += 0.25
        elif r.get("new_recipient"): s += 0.10
        if r.get("iban_country_mismatch"): s += 0.10
        s += min(float(r.get("text_risk_score", 0)) / 5.0, 0.15)
        if r.get("balance_negative"): s += 0.15
        if r.get("is_new_sender"): s += 0.05
        return float(np.clip(s, 0, 1))
