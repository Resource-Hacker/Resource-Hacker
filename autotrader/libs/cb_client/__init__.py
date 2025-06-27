"""
Minimal Coinbase Advanced-Trade + Derivatives REST client (async).
Only the endpoints we actually use.
"""
from __future__ import annotations
import httpx, time, hmac, hashlib, base64, ujson as json
from typing import Dict, Any
from decimal import Decimal
import os

BASE = "https://api-prime.coinbase.com"
if os.getenv("CB_SANDBOX","1") == "1":
    BASE = "https://sandbox-prime.coinbase.com"

class CBRest:
    def __init__(self):
        self.key        = os.environ["CB_API_KEY"]
        self.secret     = os.environ["CB_API_SECRET"]
        self.passphrase = os.environ["CB_API_PASSPHRASE"]
        self.cli        = httpx.AsyncClient(base_url=BASE, timeout=10.0)

    async def _req(self,m:str,p:str,b:Dict|None=None)->dict:
        ts=str(int(time.time()))
        body=json.dumps(b) if b else ""
        msg=f"{ts}{m.upper()}{p}{body}"
        sig=hmac.new(base64.b64decode(self.secret),
                     msg.encode(),hashlib.sha256).digest()
        h={
            "CB-ACCESS-KEY":self.key,
            "CB-ACCESS-PASSPHRASE":self.passphrase,
            "CB-ACCESS-TIMESTAMP":ts,
            "CB-ACCESS-SIGN":base64.b64encode(sig).decode(),
            "Content-Type":"application/json"
        }
        r=await self.cli.request(m,p,headers=h,content=body)
        r.raise_for_status()
        return r.json()

    # public helpers
    async def accounts(self):  return (await self._req("GET","/v3/brokerage/accounts"))["accounts"]
    async def positions(self): return (await self._req("GET","/derivatives/v3/positions"))["positions"]
    async def place(self,b):   return await self._req("POST","/v3/brokerage/orders",b)
    async def cancel_all(self):return await self._req("DELETE","/v3/brokerage/orders/batch_cancel",{})
