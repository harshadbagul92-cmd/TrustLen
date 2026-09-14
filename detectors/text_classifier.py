import os
import json
import requests
from typing import Dict, Any, List, Optional
from schemas import Signal

ZERO_SHOT_SYSTEM_PROMPT = """You are Truth Lens Text Classifier.
Analyze the user message alongside the provided heuristic rule hits.
Classify the text into EXACTLY ONE of these categories:
- "scam": Financial scams, lottery framing, prize claims, gift card demands, wire requests.
- "phishing": Credential harvesting, fake login links, OTP/PIN/password theft, account suspension threats.
- "misinformation": Unverified medical/conspiracy claims, false news, factual fabrications.
- "safe": Normal legitimate communication, announcements, assignment notices, friendly text.

Return your response in STRICT JSON format:
{
  "category": "scam" | "phishing" | "misinformation" | "safe",
  "confidence": 0.0 to 1.0,
  "reasoning_signals": [
    {
      "name": "llm_finding_name",
      "human": "Plain English description of why this was flagged",
      "weight": 0.0 to 1.0,
      "where": "Line or section description"
    }
  ]
}"""


class LLMTextClassifier:
    """
    Member A - Zero-Shot LLM Text Classifier.
    Combines input message and heuristic rule hits into structured LLM prompt, returning strict JSON.
    """

    def classify(self, text: str, rule_hits: List[Signal], api_key: Optional[str] = None) -> Dict[str, Any]:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

        if key:
            rule_summary = "\n".join([f"- {s.name}: {s.human} ({s.where})" for s in rule_hits]) if rule_hits else "None"
            prompt = f"MESSAGE TO ANALYZE:\n\"\"\"{text}\"\"\"\n\nRULE HITS DETECTED:\n{rule_summary}"
            
            try:
                if key.startswith("AIza") or os.getenv("GEMINI_API_KEY") == key:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
                    full_prompt = f"{ZERO_SHOT_SYSTEM_PROMPT}\n\n{prompt}"
                    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
                    resp = requests.post(url, json=payload, timeout=6)
                    if resp.status_code == 200:
                        raw_txt = resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        cleaned = re.sub(r"^```json\s*", "", raw_txt.strip(), flags=re.MULTILINE)
                        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
                        return json.loads(cleaned)
                else:
                    url = "https://api.openai.com/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                    payload = {
                        "model": "gpt-4o-mini",
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1
                    }
                    resp = requests.post(url, headers=headers, json=payload, timeout=6)
                    if resp.status_code == 200:
                        return json.loads(resp.json()["choices"][0]["message"]["content"])
            except Exception as e:
                print(f"[LLMTextClassifier Warning] {e}. Falling back to zero-shot rule synthesis.")

        # Zero-shot deterministic fallback classification based on rule hits
        rule_names = [s.name for s in rule_hits]
        
        if any(r in rule_names for r in ["rule_otp_password_request", "rule_punycode_digitswap_domain", "rule_display_name_mismatch", "rule_reply_to_mismatch"]):
            category = "phishing"
            confidence = 0.90
        elif any(r in rule_names for r in ["rule_payment_giftcard_demand", "rule_prize_lottery_framing", "rule_urgency_language", "rule_account_closure_threat"]):
            category = "scam"
            confidence = 0.88
        elif any(k in text.lower() for k in ["5g causes", "secret vaccine chip", "miracle cure", "aliens landed"]):
            category = "misinformation"
            confidence = 0.82
        else:
            category = "safe"
            confidence = 0.15

        return {
            "category": category,
            "confidence": confidence,
            "reasoning_signals": [
                {
                    "name": f"zero_shot_{category}_pattern",
                    "human": f"Zero-shot classification identified pattern matching {category} intent.",
                    "weight": confidence,
                    "where": "Full Document"
                }
            ]
        }
