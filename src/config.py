from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "complaints.csv"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

SAMPLE_PARQUET = PROCESSED / "sample.parquet"
SPLITS_PARQUET = PROCESSED / "splits.parquet"

SEED = 42
PER_CLASS = 6000
MIN_CHARS = 50
MIN_CLASS_ROWS = 2000

# CFPB renamed products over the years. Collapse the variants.
PRODUCT_MAP = {
    "Credit reporting": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit card": "Credit card",
    "Credit card or prepaid card": "Credit card",
    "Prepaid card": "Credit card",
    "Bank account or service": "Bank account",
    "Checking or savings account": "Bank account",
    "Payday loan": "Payday / title / personal loan",
    "Payday loan, title loan, or personal loan": "Payday / title / personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday / title / personal loan",
    "Consumer Loan": "Vehicle or consumer loan",
    "Vehicle loan or lease": "Vehicle or consumer loan",
    "Money transfer, virtual currency, or money service": "Money transfer / virtual currency",
    "Money transfers": "Money transfer / virtual currency",
    "Virtual currency": "Money transfer / virtual currency",
    "Student loan": "Student loan",
    "Mortgage": "Mortgage",
    "Debt collection": "Debt collection",
    "Debt or credit management": "Debt collection",
}