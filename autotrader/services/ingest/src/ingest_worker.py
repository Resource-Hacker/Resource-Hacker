import asyncio, json, os, time, httpx
from kafka import KafkaProducer
from libs.market import market_snapshot

BOOTSTRAP=os.getenv("KAFKA_BOOTSTRAP","localhost:9092")
producer=KafkaProducer(bootstrap_servers=[BOOTSTRAP],value_serializer=lambda v: json.dumps(v).encode())

async def get_news():
    async with httpx.AsyncClient(timeout=5) as cli:
        r=await cli.get("https://cryptopanic.com/api/v1/posts/?public=true&kind=news&regions=en")
        r.raise_for_status()
        posts=r.json().get("results",[])[:20]
        return [p["title"] for p in posts]

async def loop():
    while True:
        headlines=await get_news()
        metrics = await market_snapshot(["BTCUSDT","ETHUSDT","SOLUSDT"])
        payload={"ts":int(time.time()),"news":headlines,"metrics":metrics}
        producer.send("market.raw",payload)
        await asyncio.sleep(60)

asyncio.run(loop())
