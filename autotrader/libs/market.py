import httpx, asyncio
from typing import Dict, List, Tuple

BINANCE_SPOT_TICKER_URL="https://api.binance.com/api/v3/ticker/24hr"
BINANCE_FUTURES_PREMIUM_URL="https://fapi.binance.com/fapi/v1/premiumIndex"

async def _fetch_json(url:str, params:Dict[str,str]|None=None, timeout:int=5):
    async with httpx.AsyncClient(timeout=timeout) as cli:
        r=await cli.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def get_ticker(symbol:str="BTCUSDT")->Dict[str, str]:
    """Return 24h ticker stats for the given Binance symbol (spot)."""
    return await _fetch_json(BINANCE_SPOT_TICKER_URL, {"symbol":symbol})

async def get_funding(symbol:str="BTCUSDT")->Dict[str,str]:
    """Return current funding rate & related info for Binance perpetual futures symbol."""
    return await _fetch_json(BINANCE_FUTURES_PREMIUM_URL, {"symbol":symbol})

async def market_snapshot(symbols:List[str]|None=None)->Dict[str,Dict[str,str]]:
    """Fetch ticker and funding data concurrently for a list of symbols. Defaults to ["BTCUSDT","ETHUSDT"]."""
    symbols = symbols or ["BTCUSDT","ETHUSDT"]
    tasks: List[Tuple[str,asyncio.Task]] = []
    for sym in symbols:
        tasks.append((sym, asyncio.create_task(get_ticker(sym))))
        tasks.append((sym+"_funding", asyncio.create_task(get_funding(sym))) )
    results:Dict[str,Dict[str,str]] = {}
    for key, task in tasks:
        try:
            results[key] = await task
        except Exception as e:
            results[key] = {"error":str(e)}
    return results