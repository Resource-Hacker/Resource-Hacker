import os, asyncio, json, time
from kafka import KafkaConsumer, KafkaProducer
import openai
from config import settings
from libs.rag import add as rag_add

openai.api_key=settings.openai_api_key
consumer=KafkaConsumer("market.raw",bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP","localhost:9092")],
                       value_deserializer=lambda m: json.loads(m.decode()))
producer=KafkaProducer(bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP","localhost:9092")],
                       value_serializer=lambda v: json.dumps(v).encode())

SYSTEM="You are a senior crypto macro-analyst. Summarise docs into \u22642500-token bullet list."

async def summarise(docs):
    if not docs: return "No data."
    prompt=SYSTEM+"\n\n"+ "\n\n".join(docs)
    r=await openai.ChatCompletion.acreate(model=settings.gpt_big_model,
                                          temperature=0.1,
                                          messages=[{"role":"system","content":prompt}])
    summary=r.choices[0].message.content.strip()
    rag_add([summary])
    return summary

async def loop():
    for msg in consumer:
        docs=msg.value.get("news",[])
        summary=await summarise(docs)
        producer.send("macro.summary",{"ts":int(time.time()),"summary":summary})

asyncio.run(loop())
