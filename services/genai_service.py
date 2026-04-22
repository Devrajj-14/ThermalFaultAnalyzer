"""
GenAI summary service using Google Gemini API.
All prompts are strictly grounded in actual computed prediction data.
No hallucinated diagnoses, temperatures, or component failures.

Falls back to deterministic template-based summaries if:
- GEMINI_API_KEY is not set
- API call fails for any reason
"""

import logging
from typing import Dict

from config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)


def _template_summary(payload: Dict) -> Dict[str, str]:
    """
    Deterministic fallback summary. Concise, non-repetitive.
    All values come from the single payload dict (single source of truth).
    """
    fault_type = payload.get("fault_type", "unknown")
    severity = payload.get("severity", "unknown")
    confidence = payload.get("confidence", 0.0)
    panel_health = payload.get("panel_health", "unknown")
    action_timeline = payload.get("action_timeline", "N/A")
    area = payload.get("hotspot_area_percent", 0.0)
    regions = payload.get("region_count", 0)
    contrast = payload.get("thermal_contrast", 0.0)
    power_drop = payload.get("power_drop_estimate", 0.0)

    fault_display = fault_type.replace("_", " ").title()

    if fault_type == "normal":
        user_summary = (
            f"Thermal analysis shows no significant anomalies ({confidence*100:.0f}% confidence). "
            f"Panel health is {panel_health}. Continue routine monitoring."
        )
        maintenance_advice = (
            "No immediate action required. Maintain your regular 30-day inspection schedule."
        )
        technical_summary = (
            f"No fault detected. Anomaly area: {area:.1f}%, thermal contrast: {contrast:.1f}."
        )
    else:
        # User summary: one sentence on what was found, one on what to do
        user_summary = (
            f"A {severity}-severity {fault_display.lower()} was detected covering {area:.1f}% of the panel "
            f"({regions} region{'s' if regions != 1 else ''}, {confidence*100:.0f}% confidence). "
            f"Panel health: {panel_health}. {action_timeline}."
        )

        # Maintenance: specific, no repetition of area
        if severity == "low":
            maintenance_advice = (
                f"Inspect during the next scheduled maintenance window. "
                "Check for soiling, partial shading, or early cell degradation."
            )
        elif severity == "medium":
            maintenance_advice = (
                f"Schedule an inspection within the recommended timeframe. "
                "Investigate affected cells for damage, bypass diode issues, or soiling. "
                f"Estimated power impact: ~{power_drop:.1f}%."
            )
        else:  # high
            maintenance_advice = (
                f"Prompt inspection required. Estimated power impact: ~{power_drop:.1f}%. "
                "Check for cell damage, electrical faults, or shading. "
                "Consider isolating this panel until inspected."
            )

        # Technical: one line, all key numbers, no repetition
        technical_summary = (
            f"{fault_display} | {severity.title()} severity | {confidence*100:.0f}% confidence | "
            f"Area: {area:.1f}% | Regions: {regions} | "
            f"Thermal contrast: {contrast:.1f} | Est. power impact: {power_drop:.1f}%"
        )

    return {
        "user_summary": user_summary,
        "maintenance_advice": maintenance_advice,
        "technical_summary": technical_summary,
        "source": "template",
    }


def _gemini_summary(payload: Dict) -> Dict[str, str]:
    """
    Generate grounded summary using Google Gemini API.
    
    The prompt strictly instructs the model to use only supplied fields.
    """
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        raise RuntimeError("google-genai package not installed. Run: pip install google-genai")
    
    fault_type = payload.get("fault_type", "unknown")
    severity = payload.get("severity", "unknown")
    confidence = payload.get("confidence", 0.0)
    panel_health = payload.get("panel_health", "unknown")
    action_timeline = payload.get("action_timeline", "N/A")
    risk_timeline = payload.get("risk_timeline", "N/A")
    area = payload.get("hotspot_area_percent", 0.0)
    regions = payload.get("region_count", 0)
    temp_delta = payload.get("estimated_temp_delta", 0.0)
    power_drop = payload.get("power_drop_estimate", 0.0)
    
    prompt = f"""You are a solar panel maintenance expert AI assistant.
You have been given structured analysis results from a thermal imaging system.

STRICT RULES:
- Use ONLY the data fields provided below. Do NOT invent causes, temperatures, component names, or exact remaining useful life.
- Do NOT mention specific component failures unless explicitly stated in the data.
- Keep language professional, clear, and concise.
- Do NOT add disclaimers or caveats beyond what is relevant.

ANALYSIS DATA:
- Fault Type: {fault_type}
- Severity: {severity}
- Confidence: {confidence*100:.0f}%
- Panel Health: {panel_health}
- Hotspot Area: {area:.1f}%
- Fault Regions Detected: {regions}
- Estimated Temperature Delta: {temp_delta:.1f}°C above baseline
- Estimated Power Drop: {power_drop:.1f}%
- Action Timeline: {action_timeline}
- Risk Timeline: {risk_timeline}

Generate three outputs in this exact JSON format (no markdown, just JSON):
{{
  "user_summary": "2-3 sentence plain-language summary for a non-technical solar farm operator",
  "maintenance_advice": "2-4 sentence specific maintenance recommendation based only on the data above",
  "technical_summary": "1-2 sentence technical summary for an engineer"
}}"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    text = response.text.strip()
    
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    import json
    parsed = json.loads(text)
    parsed["source"] = "gemini"
    return parsed


def generate_summary(payload: Dict) -> Dict[str, str]:
    """
    Generate GenAI summary. Falls back to template if API unavailable.
    
    Args:
        payload: dict containing all prediction fields needed for grounding
    
    Returns:
        dict with user_summary, maintenance_advice, technical_summary, source
    """
    if not GEMINI_API_KEY:
        logger.info("GEMINI_API_KEY not set — using template summary.")
        return _template_summary(payload)
    
    try:
        result = _gemini_summary(payload)
        logger.info("Gemini summary generated successfully.")
        return result
    except Exception as exc:
        logger.warning("Gemini API call failed (%s). Falling back to template.", exc)
        result = _template_summary(payload)
        result["source"] = "template_fallback"
        return result
