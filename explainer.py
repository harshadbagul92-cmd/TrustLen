import os
import json
import requests
from typing import Optional
from schemas import Evidence, VerdictEnum

EXPLAINER_PROMPT_TEMPLATE = """You are Truth Lens, an AI forensic classifier and explainer.
Your job is to explain why an item was classified into a specific verdict band using ONLY the grounded evidence signals provided below.

STRICT GROUNDING RULES:
1. You must reference ONLY the signals present in the input JSON.
2. NEVER invent extra details, external facts, or ungrounded assumptions.
3. For "likely_authentic", phrase the summary as "no evidence of manipulation found", NEVER as "this is real" or "100% authentic".
4. Speak in clear, plain, non-technical English suitable for everyday users.

INPUT EVIDENCE:
Modality: {modality}
Verdict: {verdict}
Calibrated Score: {score} (Confidence Interval: {confidence_low} - {confidence_high})
Reliability Flags: {ood_flags}

SIGNALS DETECTED:
{signals_text}

PROVENANCE:
{provenance_text}

Generate a 2-4 sentence plain-English explanation summarizing the key signals and why they led to this verdict:"""


def generate_mock_explanation(evidence: Evidence) -> str:
    """Fallback explanation generator when LLM API key is not provided."""
    modality_name = evidence.modality.value.capitalize()
    verdict = evidence.verdict

    if verdict == VerdictEnum.DECLARED_AI:
        prov_info = ", ".join(f"{k}: {v}" for k, v in evidence.provenance.items()) if evidence.provenance else "digital metadata"
        return f"This {evidence.modality.value} contains explicit digital provenance tags ({prov_info}) declaring it as AI-generated media."

    if verdict == VerdictEnum.LIKELY_AUTHENTIC:
        if not evidence.signals:
            return f"No evidence of manipulation was found across the analyzed parameters of this {evidence.modality.value}."
        signal_summaries = " ".join([s.human for s in evidence.signals])
        return f"Analysis shows no evidence of manipulation in this {evidence.modality.value}. Key findings: {signal_summaries}"

    if verdict == VerdictEnum.LIKELY_MANIPULATED:
        signal_list = [f"- {s.human}" + (f" (Location: {s.where})" if s.where else "") for s in evidence.signals]
        signals_str = "; ".join(signal_list) if signal_list else "Multiple anomaly indicators detected."
        return (
            f"This {evidence.modality.value} exhibits strong indicators of artificial manipulation "
            f"(suspicion score {evidence.score:.2f}). Specific signals detected: {signals_str}."
        )

    # VerdictEnum.UNCERTAIN
    reasons = []
    if evidence.reliability.ood_flags:
        reasons.append(f"reliability warnings ({', '.join(evidence.reliability.ood_flags)})")
    if 0.35 <= evidence.score <= 0.65:
        reasons.append("borderline confidence scores")
    
    reason_str = " and ".join(reasons) if reasons else "mixed analytical signals"
    return (
        f"The system is uncertain about this {evidence.modality.value} due to {reason_str}. "
        f"The confidence interval spans from {evidence.confidence[0]:.2f} to {evidence.confidence[1]:.2f}. "
        f"Manual verification is recommended."
    )


def explain(
    evidence: Evidence,
    api_key: Optional[str] = None,
    provider: str = "openai"
) -> str:
    """
    Shared explainer function.
    Calls LLM if API key is supplied; otherwise returns a grounded mock explanation.
    """
    # Check environment variable if key not passed directly
    key = api_key or os.getenv("TRUTH_LENS_LLM_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not key:
        return generate_mock_explanation(evidence)

    # Format grounded prompt
    signals_text = "\n".join([
        f"- [{s.name}] Weight: {s.weight}, Value: {s.value}, Location/Where: {s.where}\n  Description: {s.human}"
        for s in evidence.signals
    ]) if evidence.signals else "None"

    provenance_text = json.dumps(evidence.provenance, indent=2) if evidence.provenance else "None"

    prompt = EXPLAINER_PROMPT_TEMPLATE.format(
        modality=evidence.modality.value,
        verdict=evidence.verdict.value,
        score=evidence.score,
        confidence_low=evidence.confidence[0],
        confidence_high=evidence.confidence[1],
        ood_flags=", ".join(evidence.reliability.ood_flags) if evidence.reliability.ood_flags else "None",
        signals_text=signals_text,
        provenance_text=provenance_text
    )

    try:
        if provider.lower() == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                return res_json['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            # Default OpenAI REST format
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are Truth Lens, an AI forensic classifier and explainer."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                return res_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Explainer LLM Error] {e}. Falling back to grounded mock explainer.")

    return generate_mock_explanation(evidence)
