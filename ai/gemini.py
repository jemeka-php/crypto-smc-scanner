"""
Optional Google Gemini setup explanation module.
Interprets deterministically calculated candidate data without hallucinating or inventing missing metrics.
"""

import json
import logging
import os
from typing import Any, Dict, Optional
from scanner.candidates import Candidate

logger = logging.getLogger(__name__)


def explain_candidate_setup(candidate: Candidate, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Send the candidate's calculated evidence to Gemini for strategic synthesis.

    Expected output format:
    {
      "assessment": "STRONG|MODERATE|WEAK|NO_TRADE",
      "bias": "LONG|SHORT|NEUTRAL",
      "reasoning": "Brief explanation.",
      "strengths": [],
      "risks": [],
      "missing_confirmation": []
    }
    """
    active_key = api_key or os.getenv("GEMINI_API_KEY", "")

    # If no key, provide deterministic analysis fallback
    if not active_key:
        return _fallback_explanation(candidate, "No GEMINI_API_KEY provided in .env or settings.")

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=active_key)

        prompt = f"""You are an elite quantitative Smart Money Concepts (SMC) analyst.
Review the following DETERMINISTIC metrics calculated by the personal scanner for {candidate.symbol}.

DO NOT recalculate or invent any numbers. Rely strictly on the supplied observations:
- Last Price: {candidate.last_price}
- 24H Price Change: {candidate.price_change_24h_pct:+.2f}%
- 24H Quote Volume: ${candidate.quote_volume_24h:,.0f}
- 1H RVOL: {candidate.rvol:.2f}x
- 1H OI Change: {candidate.oi_metrics.change_1h_pct:+.2f}%
- 4H OI Change: {candidate.oi_metrics.change_4h_pct:+.2f}%
- Price/OI Classification: {candidate.oi_metrics.classification}
- 4H Market Bias: {candidate.structure_4h.bias}
- 1H Market Bias: {candidate.structure_1h.bias}
- HTF Location: {candidate.htf_location} ({candidate.htf_location_desc})
- Recent Liquidity Sweeps: {[s.sweep_type for s in candidate.sweeps_1h]}
- Nearby Liquidity Levels: {[f"{l.source} ({l.side}) at {l.price}" for l in candidate.nearby_liquidity[:3]]}
- Active Order Blocks (1H): {len(candidate.active_obs_1h)}
- Active FVGs (1H): {len(candidate.active_fvgs_1h)}
- Candidate Scanner Stage: {candidate.stage}
- Candidate Score: {candidate.total_score}/100
- Stated Waiting For: {candidate.waiting_for}

Respond with valid JSON ONLY matching this schema:
{{
  "assessment": "STRONG" | "MODERATE" | "WEAK" | "NO_TRADE",
  "bias": "LONG" | "SHORT" | "NEUTRAL",
  "reasoning": "<Concise 2-3 sentence strategic interpretation>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "risks": ["<risk 1>", "<risk 2>"],
  "missing_confirmation": ["<missing confirmation 1>", "<missing confirmation 2>"]
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        parsed = json.loads(content.strip())
        return parsed

    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return _fallback_explanation(candidate, f"Gemini API call error: {str(e)}")


def _fallback_explanation(candidate: Candidate, note: str = "") -> Dict[str, Any]:
    """Deterministic rule-based explanation when Gemini is unavailable or not configured."""
    bias = "LONG" if candidate.direction == "BULLISH" else ("SHORT" if candidate.direction == "BEARISH" else "NEUTRAL")

    if candidate.stage == "CONFIRMED":
        assessment = "STRONG"
    elif candidate.stage == "SETUP_FORMING":
        assessment = "MODERATE"
    elif candidate.stage == "WATCH":
        assessment = "WEAK"
    else:
        assessment = "NO_TRADE"

    strengths = []
    if candidate.rvol >= 1.5:
        strengths.append(f"Elevated volume participation ({candidate.rvol:.2f}x RVOL)")
    if candidate.oi_metrics.change_1h_pct >= 5.0:
        strengths.append(f"Open interest expansion ({candidate.oi_metrics.change_1h_pct:+.2f}%)")
    if candidate.htf_location in ("NEAR DEMAND", "NEAR SUPPLY"):
        strengths.append(f"Favorable higher-timeframe confluence ({candidate.htf_location})")
    if candidate.sweeps_1h:
        strengths.append("Confirmed sell-side/buy-side liquidity sweep")

    risks = []
    if candidate.structure_4h.bias != candidate.structure_1h.bias and candidate.structure_4h.bias != "NEUTRAL":
        risks.append(f"Timeframe conflict: 4H {candidate.structure_4h.bias} vs 1H {candidate.structure_1h.bias}")
    if candidate.oi_metrics.classification.startswith("OI CONTRACTION"):
        risks.append("OI contraction suggests positioning is unwinding rather than expanding")
    if not candidate.active_obs_1h:
        risks.append("No active unmitigated order block on 1H timeframe")

    missing = [candidate.waiting_for]

    reasoning = (
        f"{candidate.symbol} is currently in stage '{candidate.stage}' with a score of {candidate.total_score}/100. "
        f"It displays {candidate.oi_metrics.classification} with {candidate.rvol:.2f}x RVOL and {candidate.htf_location} context. "
        + (f"Note: {note}" if note else "")
    )

    return {
        "assessment": assessment,
        "bias": bias,
        "reasoning": reasoning.strip(),
        "strengths": strengths or ["High liquidity perpetual futures market"],
        "risks": risks or ["Requires manual chart inspection before consideration"],
        "missing_confirmation": missing,
    }
