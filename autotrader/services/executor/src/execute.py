import os, json, asyncio, logging
from kafka import KafkaConsumer
from libs.cb_client import CBRest
from libs.models import PlanPayload

logging.basicConfig(level="INFO",format="%(asctime)s %(levelname)s %(message)s")
log=logging.getLogger("executor")
cb=CBRest()
consumer=KafkaConsumer("strategy.plan",bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP","localhost:9092")],
                       value_deserializer=lambda m: json.loads(m.decode()))

async def handle(plan):
    p=PlanPayload.model_validate(plan)
    for o in p.orders:
        body={"client_order_id":o["client_order_id"],"product_id":o["product_id"],
              "side":o["side"],"order_configuration":{"market_market_ioc":{"base_size":str(o["size_contracts"])}}}
        if o["order_type"]=="LIMIT":
            body["order_configuration"]={"limit_limit_gtc":{"base_size":str(o["size_contracts"]),
                                                            "limit_price":str(o["limit_price_usd"]),"post_only":False}}
        try:
            await cb.place(body)
            log.info("Placed %s",o["client_order_id"])
        except Exception as e:
            log.error("Order error %s",e)

async def loop():
    for msg in consumer:
        await handle(msg.value)

asyncio.run(loop())
