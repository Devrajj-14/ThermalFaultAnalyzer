# ThermalFaultAnalyzer

An intelligent solar panel thermal fault analysis system. Upload a thermal image and get multi-class fault classification, severity scoring, explainable AI heatmaps, GenAI-generated summaries, and recommended action timelines.

---

## Features

- **Multi-class classification** — normal, hotspot, severe_thermal_anomaly
- **Severity scoring** — low, medium, high
- **Panel health status** — healthy, watchlist, degraded, critical
- **XAI heatmap** — visual overlay highlighting detected fault regions with region labels
- **Grounded explanations** — every explanation point is derived from actual computed features
- **GenAI summaries** — Gemini-powered user summary, maintenance advice, and technical summary (falls back to templates if API unavailable)
- **Timeline recommendations** — action and risk timelines based on fault type and severity
- **Performance metrics** — power drop, performance ratio, degradation rate, temperature loss
- **Dual inference mode** — model-based (when trained model available) or rule-based fallback
- **Backward compatible** — legacy API fields preserved

---

## Architecture

```
ThermalFaultAnalyzer/
├── app_v2.py                  # Flask entry point (production refactor)
├── app.py                     # Original app (preserved)
├── config/
│   └── settings.py            # All constants, thresholds, env vars
├── services/
│   ├── inference_service.py   # Orchestrates model vs rule-based inference
│   ├── rule_based_service.py  # Deterministic thermal feature extraction + classification
│   ├── xai_service.py         # XAI heatmap generation + explanation points
│   ├── timeline_service.py    # Action/risk timeline recommendations
│   └── genai_service.py       # Gemini GenAI summaries with template fallback
├── utils/
│   ├── image_utils.py         # Image encoding, cropping
│   ├── validation.py          # Upload validation, filename sanitization
│   └── response_formatter.py  # Standardised JSON response builder
├── static/
│   └── index.html             # Dark-theme dashboard UI
├── models/
│   ├── convgnn_model.py       # GNN model architecture
│   └── train.py               # Training script
├── tests/
│   ├── test_timeline_service.py
│   ├── test_rule_based_service.py
│   ├── test_response_formatter.py
│   ├── test_genai_service.py
│   └── test_app_routes.py
├── data/
│   └── thermal_images/        # C (clean), H (hotspot), S (severe) images
├── .env.example
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Devrajj-14/ThermalFaultAnalyzer.git
cd ThermalFaultAnalyzer

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (optional)

python app_v2.py
```

Open **http://127.0.0.1:5001**

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Optional | Google Gemini API key for AI summaries. Falls back to templates if not set. |
| `PORT` | Optional | Server port (default: 5001) |
| `FLASK_ENV` | Optional | `development` or `production` |

---

## API Response Format

```json
{
  "success": true,
  "fault_type": "hotspot",
  "severity": "high",
  "confidence": 0.86,
  "panel_health": "degraded",
  "score": 78.4,
  "inference_mode": "rule_based",
  "action_timeline": "Inspect within 2–5 days",
  "risk_timeline": "May reduce performance in 1–3 months",
  "metrics": {
    "hotspot_area_percent": 8.4,
    "region_count": 2,
    "estimated_temp_delta": 14.2,
    "power_drop_estimate": 4.2
  },
  "xai": {
    "top_reason": "Concentrated hotspot detected covering 8.4% of panel area",
    "explanation_points": ["..."],
    "visual_path": "/static/generated/xai/xai_abc123.png",
    "xai_image_base64": "..."
  },
  "genai": {
    "user_summary": "...",
    "maintenance_advice": "...",
    "technical_summary": "...",
    "source": "gemini"
  }
}
```

---

## Inference Modes

**Rule-based (default fallback)**
Uses deterministic thresholds on extracted thermal features:
- Hotspot area percentage
- Region count
- Thermal contrast (intensity std-dev)
- Estimated temperature delta

**Model-based (future)**
The `models/` directory contains a GNN architecture. To integrate:
1. Train the model using `models/train.py` with labelled data
2. Implement `_model_based_inference()` in `services/inference_service.py`
3. The app will automatically prefer model-based inference when `models/solar_panel_gnn.pth` is present

---

## XAI Approach

This system uses transparent, region-based XAI — not a black-box model.

Every explanation is grounded in actual computed values:
- Detected hotspot regions (contours from thresholding)
- Hotspot area percentage
- Region count
- Thermal contrast between hotspot and background

A JET colormap heatmap is overlaid on the original image with bounding boxes and region labels.

---

## GenAI Grounding

The Gemini prompt strictly instructs the model:
> "Use only the supplied structured fields. Do not invent causes, temperatures, component failures, or exact remaining useful life unless explicitly provided."

If `GEMINI_API_KEY` is not set or the API call fails, a deterministic template-based summary is used automatically.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Note: If you have a broken `langsmith` pytest plugin in your environment, the `conftest.py` handles it automatically.

---

## Limitations

- **No exact remaining panel life prediction** — this requires time-series performance history and cannot be inferred from a single thermal image. The system provides recommended action timelines only.
- **Rule-based XAI** — the current XAI is based on image processing, not a trained CNN attention map. Captum/GradCAM integration is possible once a CNN model is trained.
- **Temperature proxy** — panel temperature is estimated from pixel intensity, not from a calibrated thermal sensor. Treat as a relative indicator.
- **Single image analysis** — no temporal trend analysis across multiple images.

---

## Future Improvements

- Train a CNN classifier on the C/H/S labelled dataset and integrate into `inference_service.py`
- Add Captum Integrated Gradients for true model-based XAI
- Add time-series panel performance tracking
- Add batch image analysis endpoint
- Add user authentication for multi-tenant deployments
