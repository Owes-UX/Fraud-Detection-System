import os
import requests


class LLMClient:
    def __init__(self, model: str = "openai/gpt-4.1-mini"):
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        # expects your key in env var OPENROUTER_API_KEY
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY env var is not set")

    def score_text_risk(self, text: str) -> float:
        """
        Return a risk score in [0, 1] for given text.
        0 = safe, 1 = definitely fraud.
        """

        if not text or not str(text).strip():
            return 0.0

        system_msg = (
            "You are an expert fraud analyst specializing in phishing, scams, and social engineering.\n"
            "Carefully analyze the given text for signs of fraud such as urgency, threats, fake links, "
            "account verification requests, or unusual instructions.\n"
            "Return ONLY a number between 0 and 1:\n"
            "- 0 = completely safe\n"
            "- 1 = definitely fraud\n"
        )

        user_msg = (
            "Analyze the following message:\n\n"
            f"{text}\n\n"
            "Indicators to consider:\n"
            "- urgency (act now, immediate action)\n"
            "- account/security warnings\n"
            "- suspicious links\n"
            "- requests for sensitive info\n\n"
            "Return ONLY a number between 0 and 1 (example: 0.87)"
        )

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                self.base_url, json=payload, headers=headers, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            raw = data["choices"][0]["message"]["content"].strip()
            value = float(raw)
        except Exception:
            # fallback: slightly risky but not 0
            value = 0.3

        # clip to [0, 1]
        if value < 0:
            value = 0.0
        if value > 1:
            value = 1.0
        return value