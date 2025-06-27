from __future__ import annotations
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat
from typing import List, Literal
import time

class OrderPlan(BaseModel):
    product_id: str
    side: Literal["BUY","SELL"]
    order_type: Literal["MARKET","LIMIT"]
    size_contracts: PositiveInt
    leverage: PositiveFloat
    limit_price_usd:  PositiveFloat | None = None
    stop_loss_usd:    PositiveFloat | None = None
    take_profit_usd:  PositiveFloat | None = None
    client_order_id:  str = Field(default_factory=lambda: f"oa-{int(time.time()*1e3)}")

class PlanPayload(BaseModel):
    orders: List[OrderPlan]
