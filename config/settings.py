"""
Central configuration and constants for ThermalFaultAnalyzer.
All thresholds, limits, and environment variables are managed here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── GenAI ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Upload limits ──────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024          # 10 MB
ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}

# ── Fault classification thresholds ───────────────────────────────────────────
# hotspot_area_percent thresholds
HOTSPOT_AREA_LOW    = 2.0    # % — below this → normal
HOTSPOT_AREA_MED    = 8.0    # % — below this → hotspot, above → severe
HOTSPOT_AREA_HIGH   = 15.0   # % — above this → severe regardless

# Intensity contrast thresholds (std-dev of grayscale in hotspot vs background)
CONTRAST_LOW  = 20.0
CONTRAST_HIGH = 45.0

# Region count thresholds
REGION_COUNT_MED  = 3
REGION_COUNT_HIGH = 6

# ── Severity score weights ─────────────────────────────────────────────────────
SEVERITY_AREA_WEIGHT      = 0.40
SEVERITY_CONTRAST_WEIGHT  = 0.30
SEVERITY_REGION_WEIGHT    = 0.30

# ── Performance metric constants ───────────────────────────────────────────────
BASE_EXPECTED_OUTPUT_KWH  = 1684.88   # kWh/year per kWp baseline
INSTALLED_POWER_KWP       = 1.0       # kWp (unit system)
IRRADIANCE_KWH_M2_DAY     = 5.5       # peak sun hours
DAYS_PER_YEAR             = 365
DEGRADATION_YEARS         = 5
STC_TEMP_C                = 25.0      # °C standard test condition
TEMP_COEFF                = -0.0045   # /°C crystalline silicon
RATED_POWER_W             = 1000.0    # W
MIN_PANEL_TEMP_C          = 25.0
MAX_PANEL_TEMP_C          = 85.0

# ── XAI ───────────────────────────────────────────────────────────────────────
XAI_OUTPUT_DIR = os.path.join("static", "generated", "xai")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
TEMP_UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
