import re
from typing import Dict

import pandas as pd

from utils.llm_client import LLMClient


class TextAgent:
    """
    Uses an LLM to score SMS and mails for fraud risk.

    - Builds mapping from user full name -> texts mentioning them
    - Calls LLM once per user and caches the score
    """

    def __init__(self, sms: pd.DataFrame, mails: pd.DataFrame):
        self.sms = sms
        self.mails = mails
        self.llm = LLMClient()
        self.cache: Dict[str, float] = {}

    def _collect_user_text(self, first_name: str, last_name: str) -> str:
        if not first_name or not last_name:
            return ""
        fn = first_name.strip()
        ln = last_name.strip()
        if not fn or not ln:
            return ""

        name_re_fn = re.compile(re.escape(fn), re.IGNORECASE)
        name_re_ln = re.compile(re.escape(ln), re.IGNORECASE)

        texts = []

        def scan_df(df: pd.DataFrame, col: str):
            if df is None or df.empty or col not in df.columns:
                return
            for _, row in df.iterrows():
                t = str(row.get(col, ""))
                if name_re_fn.search(t) and name_re_ln.search(t):
                    texts.append(t)

        scan_df(self.sms, "sms")
        scan_df(self.mails, "mail")

        if not texts:
            return ""
        # limit length so tokens stay reasonable
        return "\n\n---\n\n".join(texts[:50])

    def score_user(self, first_name: str, last_name: str) -> float:
        key = f"{first_name.strip()} {last_name.strip()}".lower()
        if not key.strip():
            return 0.0
        if key in self.cache:
            return self.cache[key]

        combined = self._collect_user_text(first_name, last_name)
        if not combined:
            self.cache[key] = 0.0
            return 0.0

        score = self.llm.score_text_risk(combined)
        self.cache[key] = score
        return score