import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-rough-spde-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Macro columns (optional, used as external forcing)
MACRO_COLUMNS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# SPDE parameters
HURST = 0.3               # Hurst exponent for fractional noise
GRID_SIZE = 64            # number of spatial discretisation points
SPATIAL_SCALE = 1.0
TIME_STEPS = 10           # number of time steps per day
DT = 0.01                 # time step size

# Neural Operator parameters
HIDDEN_CHANNELS = 32
KERNEL_SIZE = 3
N_LAYERS = 3
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32

# Rolling windows for evaluation (days)
WINDOWS = [63, 252, 504, 1008, 2016]

TOP_N = 3
