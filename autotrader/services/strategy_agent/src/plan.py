import asyncio, json, os, time
from kafka import KafkaConsumer, KafkaProducer
import openai
from config import settings
from libs.models import PlanPayload
from libs.rag import query as rag_query

openai.api_key=settings.openai_api_key
consumer=KafkaConsumer("macro.summary",bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP","localhost:9092")],
                       value_deserializer=lambda m: json.loads(m.decode()))
producer=KafkaProducer(bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP","localhost:9092")],
                       value_serializer=lambda v: json.dumps(v).encode())

FUNCTION_SPEC={"name":"propose_trades","parameters":PlanPayload.schema()}
SYS=f"You are an autonomous derivatives trader. Output JSON only, obey risk limits. Leverage ≤{settings.risk.max_leverage}."

async def plan(summary, metrics):
    user={"summary":summary,"metrics":metrics,"rag":rag_query('crypto market',3)}
    r=await openai.ChatCompletion.acreate(
        model=settings.gpt_small_model,
        temperature=0.15,
        messages=[{"role":"system","content":SYS},{"role":"user","content":json.dumps(user)}],
        functions=[FUNCTION_SPEC],function_call={"name":"propose_trades"})
    payload=json.loads(r.choices[0].message.function_call.arguments)
    PlanPayload.model_validate(payload)  # raises if bad
    return payload

async def loop():
    for msg in consumer:
        p=await plan(msg.value["summary"], msg.value.get("metrics",{}))
        producer.send("strategy.plan",{"ts":int(time.time()),**p})

asyncio.run(loop())
