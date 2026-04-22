"""
FeatureAgent: builds numeric + text features for each transaction.

Uses:
- users.json      → salary, job, home city / lat / lng
- locations.json  → last known lat/lng per city (rough distance)
- sms.json        → text context (via TextAgent + LLM)
- mails.json      → text context (via TextAgent + LLM)
"""

import math
from typing import Dict

import numpy as np
import pandas as pd

from agents.text_agent import TextAgent


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


class FeatureAgent:
    def __init__(
        self,
        users: pd.DataFrame,
        locations: pd.DataFrame,
        sms: pd.DataFrame,
        mails: pd.DataFrame,
        text_agent: TextAgent | None = None,
    ):
        self.users = users
        self.locations = locations
        self.sms = sms
        self.mails = mails
        self.text_agent = text_agent

        self._iban_to_user = self._build_iban_index(users)
        self._city_coords = self._build_city_coords(users)
        self._biotag_last_location = self._build_biotag_last_location(locations)

    @staticmethod
    def _build_iban_index(users: pd.DataFrame) -> Dict[str, dict]:
        mapping = {}
        if "user_iban" not in users.columns:
            return mapping
        for _, row in users.iterrows():
            iban = str(row["user_iban"])
            mapping[iban] = row.to_dict()
        return mapping

    @staticmethod
    def _build_city_coords(users: pd.DataFrame) -> Dict[str, tuple]:
        mapping: Dict[str, tuple] = {}
        city_col = "residence.city"
        lat_col = "residence.lat"
        lng_col = "residence.lng"
        if city_col not in users.columns:
            return mapping
        for _, row in users.iterrows():
            city = row.get(city_col)
            lat = row.get(lat_col)
            lng = row.get(lng_col)
            if pd.notna(city) and pd.notna(lat) and pd.notna(lng):
                mapping[str(city)] = (float(lat), float(lng))
        return mapping

    @staticmethod
    def _build_biotag_last_location(locations: pd.DataFrame) -> Dict[str, tuple]:
        mapping: Dict[str, tuple] = {}
        if locations is None or locations.empty:
            return mapping
        for _, row in locations.sort_values("timestamp").iterrows():
            biotag = row.get("biotag")
            lat = row.get("lat")
            lng = row.get("lng")
            if pd.notna(biotag) and pd.notna(lat) and pd.notna(lng):
                mapping[str(biotag)] = (float(lat), float(lng))
        return mapping

    def _get_user_info_from_iban(self, sender_iban: str):
        return self._iban_to_user.get(sender_iban)

    def extract(self, tx: pd.Series) -> dict:
        """
        Build numeric + text features for a single transaction.
        """
        feats: dict = {}

        # --- basic numeric ---
        amount = float(tx["amount"])
        balance = float(tx["balance_after"])
        feats["amount"] = amount
        feats["log_amount"] = float(np.log1p(amount))
        feats["balance"] = balance
        feats["balance_negative"] = int(balance < 0)

        # --- time features ---
        ts = tx["timestamp"]
        if pd.notna(ts):
            feats["hour"] = int(ts.hour)
            feats["weekday"] = int(ts.weekday())
        else:
            feats["hour"] = 12
            feats["weekday"] = 0

        # --- type / payment method encodings ---
        tx_type = str(tx["transaction_type"]).lower()
        pm = str(tx["payment_method"]).lower()

        type_map = {
            "transfer": 0,
            "withdrawal": 1,
            "card payment": 2,
            "online_purchase": 3,
        }
        feats["tx_type_id"] = type_map.get(tx_type, -1)

        pm_map = {
            "debit_card": 0,
            "credit_card": 1,
            "paypal": 2,
            "mobile_wallet": 3,
            "bank_transfer": 4,
        }
        feats["pay_method_id"] = pm_map.get(pm.replace(" ", "_"), -1)

        # --- IBAN country mismatch ---
        s_iban = str(tx["sender_iban"])
        r_iban = str(tx["recipient_iban"])
        s_country = s_iban[:2] if len(s_iban) >= 2 else "XX"
        r_country = r_iban[:2] if len(r_iban) >= 2 else "XX"
        feats["iban_country_mismatch"] = int(
            s_country != r_country and s_country != "XX" and r_country != "XX"
        )

        # --- user info from users.json ---
        user_info = self._get_user_info_from_iban(s_iban)
        if user_info is not None:
            salary = float(user_info.get("salary", 0.0))
            feats["salary"] = salary
            feats["amount_over_salary"] = float(amount / salary) if salary > 0 else 0.0
            home_city = str(user_info.get("residence.city", ""))
            first_name = str(user_info.get("first_name", "")).strip()
            last_name = str(user_info.get("last_name", "")).strip()
        else:
            feats["salary"] = 0.0
            feats["amount_over_salary"] = 0.0
            home_city = ""
            first_name = ""
            last_name = ""

        # --- distance: home city vs transaction city ---
        tx_city = str(tx["location"])
        same_city = bool(home_city) and bool(tx_city) and (str(home_city) == str(tx_city))
        feats["home_equals_tx_city"] = 1 if same_city else 0

        dist_km = 0.0
        if home_city in self._city_coords and tx_city in self._city_coords:
            lat1, lon1 = self._city_coords[home_city]
            lat2, lon2 = self._city_coords[tx_city]
            dist_km = haversine_km(lat1, lon1, lat2, lon2)
        feats["home_tx_distance_km"] = dist_km
        feats["far_from_home"] = int(dist_km > 300)

        # --- LLM-based text risk per user ---
        if user_info is not None and self.text_agent is not None:
            feats["text_risk_score"] = float(
                self.text_agent.score_user(first_name, last_name)
            )
        else:
            feats["text_risk_score"] = 0.0

        return feats