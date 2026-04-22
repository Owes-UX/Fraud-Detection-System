"""
DataAgent: loads and cleans all input datasets for THIS challenge.

Expected files in ./data/:

- transactions.csv   (main transactions)
- users.json         (user profiles, one big JSON array)
- locations.json     (GPS traces)
- sms.json           (SMS threads)
- mails.json         (email threads)
"""
import os
import json
import pandas as pd


class DataAgent:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.transactions: pd.DataFrame | None = None
        self.users: pd.DataFrame | None = None
        self.locations: pd.DataFrame | None = None
        self.sms: pd.DataFrame | None = None
        self.mails: pd.DataFrame | None = None

    def _csv_path(self, name: str) -> str:
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected CSV not found: {path}")
        return path

    def _json_path(self, name: str) -> str:
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected JSON not found: {path}")
        return path

    def load_transactions(self):
        # EXACT columns expected:
        # transaction_id,sender_id,recipient_id,transaction_type,amount,location,
        # payment_method,sender_iban,recipient_iban,balance_after,description,timestamp
        path = self._csv_path("transactions.csv")
        tx = pd.read_csv(path, low_memory=False)
        tx.columns = [c.strip() for c in tx.columns]

        # Basic type cleaning
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
        tx["balance_after"] = pd.to_numeric(
            tx["balance_after"], errors="coerce"
        ).fillna(0.0)
        tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")

        # Sort by time so our profile agent respects causality
        tx = tx.sort_values("timestamp").reset_index(drop=True)

        self.transactions = tx
        print(f"[DataAgent] transactions.csv loaded: {len(tx)} rows")

    def load_users(self):
        # users.json is a JSON array of objects, each with:
        # first_name,last_name,birth_year,salary,job,iban,residence{city,lat,lng},description
        path = self._json_path("users.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        # make it easy to join on IBAN and also to get coordinates
        df.rename(columns={"iban": "user_iban"}, inplace=True)
        self.users = df
        print(f"[DataAgent] users.json loaded: {len(df)} users")

    def load_locations(self):
        # locations.json: array with biotag,timestamp,lat,lng,city
        path = self._json_path("locations.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        self.locations = df.sort_values("timestamp").reset_index(drop=True)
        print(f"[DataAgent] locations.json loaded: {len(df)} rows")

    def load_sms(self):
        # sms.json: array like { "sms": "From: ... To: ... Date: ... Message: ..." }
        path = self._json_path("sms.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)  # one column: "sms"
        self.sms = df
        print(f"[DataAgent] sms.json loaded: {len(df)} messages")

    def load_mails(self):
        # mails.json: array like { "mail": "From: ... To: ... Subject: ... <HTML>" }
        path = self._json_path("mails.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)  # one column: "mail"
        self.mails = df
        print(f"[DataAgent] mails.json loaded: {len(df)} mails")

    def load(self):
        print(f"[DataAgent] Loading from {os.path.abspath(self.data_dir)}")
        self.load_transactions()
        self.load_users()
        self.load_locations()
        self.load_sms()
        self.load_mails()
        return self