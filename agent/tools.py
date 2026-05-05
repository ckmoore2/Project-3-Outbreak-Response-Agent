"""
agent/tools.py — Four LangChain tools for the Outbreak Response Agent.

Uses the real Project 2 artifacts:
  - model/szr_predictor.py  → SZRPredictor class + FEATURE_COLUMNS
  - model/best_model.pt     → Experiment D weights (R²=0.9424)
  - model/scaler.pkl        → fitted StandardScaler
  - data/nc_counties.json   → all 100 NC counties with real HSI sub-scores

FEATURE_COLUMNS (must match scaler fit order):
  ["beta", "zeta", "alpha", "initial_population", "initial_infected",
   "mobility_score", "infrastructure_score", "health_score"]

Intervention levels map to (beta, zeta) pairs drawn from the scenario
ranges used in generate_data.py, holding beta at an "active spread"
baseline and scaling zeta upward with response intensity.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from langchain_core.tools import tool

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parents[1]
MODEL_PATH  = Path(os.getenv("MODEL_PATH",  str(_ROOT / "model" / "best_model.pt")))
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(_ROOT / "model" / "scaler.pkl")))
COUNTY_JSON = _ROOT / "data" / "nc_counties.json"

# ── Intervention levels → (beta, zeta, alpha) ─────────────────────────────────
# Drawn from scenario ranges in data/generate_data.py.
# beta fixed at mid active_spread; zeta scales from collapse → military.
INTERVENTIONS: dict[str, dict] = {
    "none": {
        "beta": 0.38, "zeta": 0.09, "alpha": 0.015,
        "label": "No organised response — civilian baseline",
    },
    "low": {
        "beta": 0.35, "zeta": 0.17, "alpha": 0.015,
        "label": "Low — informal neighbourhood coordination",
    },
    "medium": {
        "beta": 0.30, "zeta": 0.32, "alpha": 0.015,
        "label": "Medium — structured local emergency response",
    },
    "high": {
        "beta": 0.22, "zeta": 0.45, "alpha": 0.015,
        "label": "High — full military / EMD deployment",
    },
}

# ── Model singleton ────────────────────────────────────────────────────────────
_model  = None
_scaler = None


def _load_artifacts():
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    sys.path.insert(0, str(_ROOT / "model"))
    from szr_predictor import SZRPredictor

    _model = SZRPredictor(
        input_dim=8,
        hidden_dims=[128, 256, 128],
        output_dim=3,
        dropout=0.2,
    )
    _model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    _model.eval()

    _scaler = joblib.load(SCALER_PATH)

    return _model, _scaler


# ── County data ────────────────────────────────────────────────────────────────
_county_db: dict | None = None


def _get_db() -> dict:
    global _county_db
    if _county_db is None:
        with open(COUNTY_JSON) as f:
            _county_db = json.load(f)
    return _county_db


def _find_county_name(name: str) -> str | None:
    db = _get_db()
    key = name.strip().title()
    if key in db:
        return key
    q = name.strip().lower()
    for k in db:
        if q in k.lower():
            return k
    return None


def _get_county(name: str) -> dict | None:
    matched = _find_county_name(name)
    return _get_db().get(matched) if matched else None


# ── Raw inference ──────────────────────────────────────────────────────────────

def _run_inference(profile: dict, intervention: str) -> dict:
    """
    Build the 8-feature vector (FEATURE_COLUMNS order) and run the model.
    Feature order: beta, zeta, alpha, initial_population, initial_infected,
                   mobility_score, infrastructure_score, health_score
    """
    model, scaler = _load_artifacts()
    iv = INTERVENTIONS[intervention]

    x_raw = pd.DataFrame([[
        iv["beta"],
        iv["zeta"],
        iv["alpha"],
        float(profile["population"]),
        1.0,
        float(profile["mobility_score"]),
        float(profile["infrastructure_score"]),
        float(profile["health_score"]),
    ]], columns=["beta", "zeta", "alpha", "initial_population", "initial_infected",
                 "mobility_score", "infrastructure_score", "health_score"])

    x_scaled = scaler.transform(x_raw)
    x_tensor  = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        out = model(x_tensor).squeeze().numpy()

    peak_frac    = float(np.clip(out[0], 0.0, 1.0))
    time_to_peak = float(np.clip(out[1], 0.0, 180.0))
    contain_prob = float(torch.sigmoid(torch.tensor(out[2])).item())

    return {
        "peak_zombie_fraction": round(peak_frac, 4),
        "time_to_peak_days":    round(time_to_peak, 1),
        "containment_prob":     round(contain_prob, 4),
        "contained":            contain_prob >= 0.5,
        "survival_fraction":    round(float(np.clip(1.0 - peak_frac, 0.0, 1.0)), 4),
    }


def _hsi_label(hsi: float) -> str:
    if hsi >= 0.55: return "High — well-organised, strong resources"
    if hsi >= 0.42: return "Medium — moderate preparedness"
    if hsi >= 0.35: return "Low — limited resources, poor coordination"
    return "Critical — highly vulnerable population"


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — County profile lookup
# ══════════════════════════════════════════════════════════════════════════════

@tool
def lookup_county_profile(county_name: str) -> str:
    """
    Look up the demographic and HSI (Human Survivability Index) profile for a
    named North Carolina county using real ACS and CDC PLACES data.

    Returns population, composite HSI score, all six HSI sub-scores (health,
    mobility, infrastructure, education, social/community, geographic),
    urbanicity, and tier classifications. Always call this before predict_outbreak
    to understand a county's baseline characteristics.

    If the county name is not found, returns a list of suggestions.

    Args:
        county_name: Name of the NC county, e.g. "Robeson", "Wake", "Pitt".
    """
    profile = _get_county(county_name)
    if profile is None:
        db = _get_db()
        q = county_name.strip().lower()
        suggestions = [k for k in sorted(db.keys()) if q in k.lower()]
        return json.dumps({
            "error":       f"County '{county_name}' not found.",
            "suggestions": suggestions[:6] if suggestions else sorted(db.keys())[:10],
        }, indent=2)

    matched = _find_county_name(county_name)
    return json.dumps({
        "county":             matched,
        "fips":               profile["fips"],
        "population":         profile["population"],
        "hsi":                profile["hsi"],
        "hsi_interpretation": _hsi_label(profile["hsi"]),
        "hsi_sub_scores": {
            "health_score":         profile["health_score"],
            "mobility_score":       profile["mobility_score"],
            "infrastructure_score": profile["infrastructure_score"],
            "education_score":      profile["education_score"],
            "social_score":         profile["social_score"],
            "geo_score":            profile["geo_score"],
        },
        "norm_pop_density":  profile["norm_pop_density"],
        "rural":             profile["rural"],
        "infrastructure_tier": profile["infrastructure_tier"],
        "mobility_tier":     profile["mobility_tier"],
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — Single scenario prediction
# ══════════════════════════════════════════════════════════════════════════════

@tool
def predict_outbreak(county_name: str, intervention_level: str = "none") -> str:
    """
    Run the SZR surrogate model (Experiment D, R²=0.9424) for a specific NC
    county under a given intervention level and return quantitative outbreak
    outcome metrics.

    Intervention levels:
        none   — β=0.38, ζ=0.09 (rapid collapse likely)
        low    — β=0.35, ζ=0.17 (informal coordination)
        medium — β=0.30, ζ=0.32 (structured local response)
        high   — β=0.22, ζ=0.45 (military/EMD, containment likely)

    HSI sub-scores (health, mobility, infrastructure) from real county data
    are passed directly as model features.

    Args:
        county_name:        NC county name, e.g. "Robeson".
        intervention_level: One of "none", "low", "medium", "high".
    """
    level = intervention_level.lower().strip()
    if level not in INTERVENTIONS:
        return json.dumps({
            "error": f"Unknown intervention level '{intervention_level}'.",
            "valid": list(INTERVENTIONS.keys()),
        })

    profile = _get_county(county_name)
    if profile is None:
        return json.dumps({"error": f"County '{county_name}' not found. Call lookup_county_profile first."})

    try:
        r = _run_inference(profile, level)
    except Exception as e:
        return json.dumps({"error": f"Inference failed: {e}"})

    matched = _find_county_name(county_name)
    return json.dumps({
        "county":            matched,
        "intervention":      level,
        "intervention_desc": INTERVENTIONS[level]["label"],
        "hsi":               profile["hsi"],
        "hsi_label":         _hsi_label(profile["hsi"]),
        "peak_zombie_pct":   f"{r['peak_zombie_fraction'] * 100:.1f}%",
        "time_to_peak_days": r["time_to_peak_days"],
        "survival_pct":      f"{r['survival_fraction'] * 100:.1f}%",
        "containment_prob":  f"{r['containment_prob'] * 100:.1f}%",
        "contained":         r["contained"],
        "raw":               r,
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 3 — Multi-scenario comparison
# ══════════════════════════════════════════════════════════════════════════════

@tool
def compare_interventions(county_name: str) -> str:
    """
    Run the SZR surrogate model across all four intervention levels for a
    given NC county and return a ranked comparison table.

    Results are sorted by survival fraction (highest first). Also reports
    the minimum intervention level that achieves predicted containment.

    Args:
        county_name: NC county name, e.g. "Robeson".
    """
    profile = _get_county(county_name)
    if profile is None:
        return json.dumps({"error": f"County '{county_name}' not found. Call lookup_county_profile first."})

    matched = _find_county_name(county_name)
    rows = []
    for level in INTERVENTIONS:
        try:
            r = _run_inference(profile, level)
            rows.append({
                "intervention":      level,
                "description":       INTERVENTIONS[level]["label"],
                "peak_zombie_pct":   round(r["peak_zombie_fraction"] * 100, 1),
                "time_to_peak_days": r["time_to_peak_days"],
                "survival_pct":      round(r["survival_fraction"] * 100, 1),
                "containment_prob_pct": round(r["containment_prob"] * 100, 1),
                "contained":         r["contained"],
            })
        except Exception as e:
            rows.append({"intervention": level, "error": str(e)})

    rows.sort(key=lambda x: x.get("survival_pct", 0), reverse=True)

    order = ["none", "low", "medium", "high"]
    min_contain = next(
        (r["intervention"] for r in sorted(rows, key=lambda x: order.index(x.get("intervention", "high")))
         if r.get("contained")),
        "No intervention achieves containment",
    )

    return json.dumps({
        "county":                matched,
        "population":            profile["population"],
        "hsi":                   profile["hsi"],
        "hsi_label":             _hsi_label(profile["hsi"]),
        "ranked_scenarios":      rows,
        "min_containment_level": min_contain,
        "key_hsi_sub_scores": {
            "health_score":         profile["health_score"],
            "mobility_score":       profile["mobility_score"],
            "infrastructure_score": profile["infrastructure_score"],
        },
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 4 — Report synthesis
# ══════════════════════════════════════════════════════════════════════════════

@tool
def synthesize_report(
    counties_analyzed: str,
    key_findings: str,
    recommended_intervention: str,
    recommendation_justification: str,
    caveats: str = "",
) -> str:
    """
    Produce a structured final recommendation report in markdown. Always call
    this as the FINAL step after all predictions and comparisons are complete.

    Args:
        counties_analyzed:            Comma-separated list of counties examined.
        key_findings:                 Most important quantitative findings,
                                      including specific percentages and day counts.
        recommended_intervention:     The recommended intervention level and counties.
        recommendation_justification: Quantitative reasoning citing model outputs.
        caveats:                      Model limitations or scenario assumptions.
    """
    counties = [c.strip().title() for c in counties_analyzed.split(",")]
    count    = len(counties)

    report = f"""# 🧟 Outbreak Response Planning Report

## Executive Summary
SZR surrogate model analysis (Experiment D — hidden=[128,256,128], R²=0.9424)
for **{', '.join(counties)}** ({count} {'county' if count == 1 else 'counties'}) across four intervention
levels using real NC county HSI profiles (ACS · CDC PLACES · USFA · FCC data).

---

## Counties Analyzed
{chr(10).join(f"- **{c}**" for c in counties)}

---

## Key Findings

{key_findings}

---

## Recommendation

**{recommended_intervention}**

{recommendation_justification}

---

## Caveats & Limitations

{caveats if caveats else ""}

- Surrogate model approximates full ODE dynamics; outputs should be validated
  against county-level tract simulations before operational use.
- HSI sub-scores from ACS/CDC PLACES; rapidly evolving outbreak conditions
  may not be reflected.
- Intervention β/ζ values represent scenario archetypes, not real-time
  resource deployment capacity.

---
*Generated by the Autonomous Outbreak Response Planning Agent*
*Model: SZRPredictor Experiment D | Framework: LangGraph ReAct*
*Data: 100 NC counties — ACS, CDC PLACES, USFA, FCC*
"""
    return report
