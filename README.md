# ThermalFaultAnalyzer

Intelligent solar panel thermal fault detection and analysis system. Upload a thermal image and get multi-class fault classification, severity scoring, explainable AI heatmaps, AI-generated summaries, and recommended action timelines — all in a clean dark-theme dashboard.
<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/e58f0d66-501e-4c13-b037-a54f70f0f74b" />


---

## Demo

### Upload & Analyze
![Upload UI](docs/screenshots/upload.png)

Drop any thermal image into the dashboard and click **Analyze Thermal Image**. Results appear instantly below.

---

### Classification Summary
![Classification Banner](docs/screenshots/classification.png)

The top banner shows the five key outputs at a glance:

| Field | Example |
|---|---|
| Fault Classification | Hotspot |
| Panel Health | Degraded |
| Severity | High |
| Confidence | 92% |
| Risk Score | 78.9 |

---

### Fault Detection & XAI Heatmap
<img width="1280" height="577" alt="image" src="https://github.com/user-attachments/assets/4fe80244-c5d8-4462-b8d9-0ec12fae2b9e" />

![Detection and XAI](docs/screenshots/detection_xai.png)

Three side-by-side panels:
- **Input Image** — original thermal image as uploaded
- **Fault Detection** — bounding boxes drawn around ranked anomaly regions
- **XAI Heatmap** — INFERNO colormap overlay showing thermal intensity distribution, with region labels and area percentages

---

### Why This Result
![Explanation](docs/screenshots/explanation.png)
<img width="1280" height="578" alt="image" src="https://github.com/user-attachments/assets/b235ec07-fd64-4573-9c26-87ebcbd5aec9" />


Concise evidence bullets grounded in computed values — no invented explanations:
- Anomaly area coverage
- Number of distinct regions
- Thermal contrast score
- Recommended action

Paired with a **Recommended Timeline** card showing action deadline and risk outlook.

---

### AI-Generated Analysis
<img width="1280" height="439" alt="image" src="https://github.com/user-attachments/assets/693ca357-b084-4197-8d23-f662c694429b" />



Three sections powered by Gemini (falls back to deterministic templates if API unavailable):
- **Summary** — plain-language overview for operators
- **Maintenance Recommendation** — specific next steps
- **Technical Summary** — one-line engineer-facing digest

---

### Technical Metrics Table
![Metrics](docs/screenshots/metrics.png)
<img width="1280" height="433" alt="image" src="https://github.com/user-attachments/assets/4f81c886-e19a-4f04-a339-c32411e512d7" />


| Metric | Est. Value | Threshold | Status |
|---|---|---|---|
| Power Impact (est.) | 17.79% | <10% | 🔴 |
| Performance Ratio | 69% | ≥75% | 🔴 |
| Annual Degradation | 3.56%/yr | <0.7%/year | 🔴 |
| Thermal Power Loss (est.) | 54.4 W | — | ⚠️ |
| Thermal Level | Elevated | — | — |

All values are estimates derived from image analysis, not calibrated sensor readings.

---

### Detected Fault Regions
![Regions](docs/screenshots/regions.png)
<img width="1280" height="675" alt="image" src="https://github.com/user-attachments/assets/0517b83b-9ce6-49fc-b225-27f1293130bb" />


Each detected region is shown as a ranked row with:
- Square-padded 128×128 thumbnail
- Area %, importance score, contrast score
- Bounding box coordinates and dimensions

---

## Features

- **Multi-class classification** — normal, hotspot, severe_thermal_anomaly
- **Severity scoring** — low, medium, high
- **Panel health status** — healthy, watchlist, degraded, critical
- **Color-aware detection** — works on JET/thermal colormap images using red-channel dominance scoring, not grayscale
- **Multi-factor region ranking** — area, peak thermal score, contrast, compactness, border penalty
- **XAI heatmap** — INFERNO overlay with ranked bounding boxes, region labels, and legend
- **Grounded explanations** — every point derived from actual computed features
- **GenAI summaries** — Gemini-powered with deterministic template fallback
- **Timeline recommendations** — proportional to fault type and severity
- **Consistent metrics** — single source of truth; all cards read from the same computed object
- **Dual inference mode** — model-based (when trained model present) or rule-based fallback

---

## Architecture

```
ThermalFaultAnalyzer/
├── app_v2.py                  # Flask entry point
├── config/
│   └── settings.py            # All constants, thresholds, env vars
├── services/
│   ├── inference_service.py   # Orchestrates model vs rule-based inference
│   ├── rule_based_service.py  # Color-aware thermal feature extraction + classification
│   ├── xai_service.py         # XAI heatmap + explanation points
│   ├── timeline_service.py    # Action/risk timeline recommendations
│   └── genai_service.py       # Gemini summaries with template fallback
├── utils/
│   ├── image_utils.py         # Image encoding, square-padded region crops
│   ├── validation.py          # Upload validation, filename sanitization
│   └── response_formatter.py  # Standardised JSON response builder
├── static/
│   └── index.html             # Dark-theme dashboard UI
├── models/
│   ├── convgnn_model.py       # GNN model architecture
│   └── train.py               # Training script
├── tests/
│   └── ...                    # pytest test suite
├── data/
│   └── thermal_images/        # Dataset (C/H/S labelled images)
├── .env.example
└── requirements.txt
```

---

## How to Start

### 1. Clone the repo

```bash
git clone https://github.com/Devrajj-14/ThermalFaultAnalyzer.git
cd ThermalFaultAnalyzer
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key if you want AI-generated summaries:

```
GEMINI_API_KEY=your_key_here
```

Leave it blank to use the deterministic template fallback — the app works fully without it.

### 5. Run the server

```bash
python app_v2.py
```

### 6. Open the dashboard

```
http://127.0.0.1:5001
```

Upload any thermal image (JPG, PNG, BMP, TIFF — max 10 MB) and click **Analyze Thermal Image**.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Optional | Google Gemini API key for AI summaries. Falls back to templates if not set. |
| `PORT` | Optional | Server port (default: 5001) |

---

## API

**POST** `/predict`

```
Content-Type: multipart/form-data
Field: image  (file)
```

Example with curl:

```bash
curl -X POST -F "image=@your_thermal_image.jpg" http://127.0.0.1:5001/predict
```

Response shape:

```json
{
  "success": true,
  "fault_type": "hotspot",
  "severity": "high",
  "confidence": 0.92,
  "panel_health": "degraded",
  "score": 78.9,
  "inference_mode": "rule_based",
  "action_timeline": "Inspect within 3–7 days",
  "risk_timeline": "May reduce performance in 1–3 months",
  "metrics": {
    "hotspot_area_percent": 12.7,
    "region_count": 5,
    "thermal_contrast": 30.3,
    "power_drop_estimate": 17.8
  },
  "xai": {
    "top_reason": "Significant thermal anomaly covering 12.7% of panel area across 5 region(s).",
    "explanation_points": ["..."],
    "xai_image_base64": "...",
    "ranked_regions": [
      { "area_pct": 3.82, "importance": 0.90, "contrast": 0.23, "x": 171, "y": 197, "bw": 80, "bh": 149 }
    ]
  },
  "genai": {
    "user_summary": "...",
    "maintenance_advice": "...",
    "technical_summary": "Hotspot | High severity | 92% confidence | Area: 12.7% | Regions: 5 | Thermal contrast: 30.3 | Est. power impact: 17.8%",
    "source": "template"
  }
}
```

**GET** `/health` — returns `{"status": "healthy"}`

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Detection Approach

The detector works on **color-mapped thermal images** (JET colormap), not raw grayscale:

1. Computes a per-pixel **thermal score** from RGB channel relationships — red minus blue captures the JET hot-to-cold axis, with boosts for orange/yellow and near-white pixels
2. Applies **adaptive percentile thresholding** (top 8% of thermal scores) with a dynamic range check for uniform images
3. Scores each candidate region by a weighted combination of: peak score (30%), mean score (25%), log-normalised area (20%), compactness (10%), interior penalty (15%)
4. **Penalises thin border-hugging strips** — regions touching the image edge with aspect ratio > 4 get a 0.1× interior score
5. **Merges nearby regions** via dilation before re-scoring
6. Keeps the **top 5 regions by importance score**

---

## Limitations

- Temperature values are visual proxies from image intensity, not calibrated sensor readings
- Rule-based XAI uses image processing, not trained model attention maps
- Single-image analysis only — no temporal trend tracking
- No exact remaining panel life prediction from a single image

---

## Future Work

- Train a CNN classifier on the C/H/S labelled dataset
- Integrate Captum Integrated Gradients for model-based XAI
- Add time-series panel performance tracking
- Add batch analysis endpoint
