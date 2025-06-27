from fastapi import FastAPI
from kafka import KafkaConsumer
import json, os

app=FastAPI()
consumer=KafkaConsumer("strategy.plan","macro.summary",bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP","localhost:9092")],
                       value_deserializer=lambda m: json.loads(m.decode()),auto_offset_reset="latest",consumer_timeout_ms=1000)

state={"last_plan":None,"last_summary":None}

@app.get("/status")
def status(): return state

for msg in consumer:
    if msg.topic=="strategy.plan": state["last_plan"]=msg.value
    if msg.topic=="macro.summary": state["last_summary"]=msg.value
