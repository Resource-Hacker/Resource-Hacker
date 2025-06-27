from pydantic import BaseModel, PositiveInt
from decimal import Decimal
import os, dotenv
dotenv.load_dotenv()

class RiskLimits(BaseModel):
    max_leverage: PositiveInt = 20
    max_orders:   PositiveInt = 10
    free_collateral_pct: Decimal = Decimal("0.10")
    drawdown_kill_pct:   Decimal = Decimal("0.35")
    notional_cap_usd:    Decimal = Decimal("250000")

class Settings(BaseModel):
    sandbox:           bool = bool(int(os.getenv("CB_SANDBOX","1")))
    cycle_seconds:     int  = 60
    cb_api_key:        str  = os.getenv("CB_API_KEY","")
    cb_api_secret:     str  = os.getenv("CB_API_SECRET","")
    cb_api_passphrase: str  = os.getenv("CB_API_PASSPHRASE","")
    openai_api_key:    str  = os.getenv("OPENAI_API_KEY","")
    gpt_big_model:     str  = "gpt-4.1"
    gpt_small_model:   str  = "o3-pro"
    risk:              RiskLimits = RiskLimits()
    vec_store_path:    str  = "rag_store"

settings = Settings()
