"""
DecisionAgent: Applies an adaptive threshold to produce final flagged IDs.
"""
import numpy as np

class DecisionAgent:
    def __init__(self, base_threshold=0.45, target_flag_rate=0.08):
        self.threshold = base_threshold
        self.target_flag_rate = target_flag_rate

    def decide(self, transaction_ids, scores):
        scores_arr = np.array(scores)
        n = len(scores_arr)
        if n == 0:
            return []
        adaptive = float(np.quantile(scores_arr, 1 - self.target_flag_rate))
        threshold = float(np.clip(0.5 * self.threshold + 0.5 * adaptive, 0.2, 0.85))
        flagged = [tid for tid, s in zip(transaction_ids, scores) if s >= threshold]
        if len(flagged) == 0:
            top_idx = np.argsort(scores_arr)[::-1][:max(1, int(n * 0.01))]
            flagged = [transaction_ids[i] for i in top_idx]
        if len(flagged) == n:
            top_idx = np.argsort(scores_arr)[::-1][:max(1, int(n * 0.15))]
            flagged = [transaction_ids[i] for i in top_idx]
        print(f"[DecisionAgent] Threshold={threshold:.3f} | Flagged {len(flagged)}/{n} ({len(flagged)/n:.1%})")
        return flagged
