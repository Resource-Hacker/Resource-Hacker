#!/usr/bin/env bash
set -euo pipefail

PROJECT="autotrader"
PY_VERSION="3.11"

echo "🥚  Initialising $PROJECT …"

# 1.  Directory tree
mkdir -p "$PROJECT"/{services/{ingest,macro_agent,strategy_agent,executor,api}/src,\
libs/{cb_client,models,rag},infra/k8s,scripts,tests}

# 2.  pyproject.toml  (single Poetry workspace)
cat > "$PROJECT/pyproject.toml" << TOML
[tool.poetry]
name = "autotrader"
version = "0.1.0"
description = "Multi-model autonomous crypto trading stack"
authors = ["Your Team <team@example.com>"]
packages = [{include = "libs"}, {include = "services"}]

[tool.poetry.dependencies]
python = "^$PY_VERSION"
pydantic = "^2.7.1"
httpx = "^0.27"
openai = "^1.23"
ujson = "^5.10.0"
fastapi = "^0.111"
uvicorn = {extras=["standard"], version="^0.29"}
kafka-python = "^2.0"
qdrant-client = "^1.8"
sentence-transformers = "^2.7"
faiss-cpu = "^1.8"
websocket-client = "^1.8"

[tool.poetry.group.dev.dependencies]
pytest = "^8.2"
ruff = "^0.4"
mypy = "^1.10"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
TOML

# 3.  .env.example  (secrets placeholder)
cat > "$PROJECT/.env.example" << 'ENV'
# Coinbase
CB_API_KEY=""
CB_API_SECRET=""
CB_API_PASSPHRASE=""
CB_SANDBOX=1

# OpenAI
OPENAI_API_KEY=""

# Optional: Kafka, Postgres, Qdrant creds …
ENV

# 4.  Shared libs  (schemas + simple cb client + RAG helpers)
cat > "$PROJECT/libs/models/__init__.py" << 'PY'
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
PY

cat > "$PROJECT/libs/cb_client/__init__.py" << 'PY'
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
PY

cat > "$PROJECT/libs/rag/__init__.py" << 'PY'
import faiss, numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384
index = faiss.IndexFlatIP(DIM)
docs: List[str] = []

def add(texts:List[str]): 
    global docs
    embs = _model.encode(texts, normalize_embeddings=True)
    index.add(np.asarray(embs,dtype="float32"))
    docs.extend(texts)

def query(q:str,k:int=5)->List[str]:
    if index.ntotal==0: return []
    emb = _model.encode([q], normalize_embeddings=True)
    D,I = index.search(np.asarray(emb,dtype="float32"),k)
    return [docs[i] for i in I[0] if i < len(docs)]
PY

# 5.  config.py  (risk + model settings)
cat > "$PROJECT/config.py" << 'PY'
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
PY

# 6.  Service implementations  (ingest, macro_agent, strategy_agent, executor, api)
# 6a.  INGEST  (price / news) – simplified worker
cat > "$PROJECT/services/ingest/src/ingest_worker.py" << 'PY'
import asyncio, json, os, time, httpx
from kafka import KafkaProducer

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
        producer.send("market.raw",{"ts":int(time.time()),"news":headlines})
        await asyncio.sleep(60)

asyncio.run(loop())
PY

cat > "$PROJECT/services/ingest/Dockerfile" << 'DOCKER'
FROM python:3.11-slim
WORKDIR /app
COPY ../../.. /app
RUN pip install --no-cache-dir -e .
CMD ["python","-m","services.ingest.src.ingest_worker"]
DOCKER

# 6b.  MACRO_AGENT  (GPT-4.1 summariser)
cat > "$PROJECT/services/macro_agent/src/summarise.py" << 'PY'
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
PY

cat > "$PROJECT/services/macro_agent/Dockerfile" << 'DOCKER'
FROM python:3.11-slim
WORKDIR /app
COPY ../../.. /app
RUN pip install --no-cache-dir -e .
CMD ["python","-m","services.macro_agent.src.summarise"]
DOCKER

# 6c.  STRATEGY_AGENT  (o3-pro planner)
cat > "$PROJECT/services/strategy_agent/src/plan.py" << 'PY'
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

async def plan(summary):
    user={"summary":summary,"rag":rag_query('crypto market',3)}
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
        p=await plan(msg.value["summary"])
        producer.send("strategy.plan",{"ts":int(time.time()),**p})

asyncio.run(loop())
PY

cat > "$PROJECT/services/strategy_agent/Dockerfile" << 'DOCKER'
FROM python:3.11-slim
WORKDIR /app
COPY ../../.. /app
RUN pip install --no-cache-dir -e .
CMD ["python","-m","services.strategy_agent.src.plan"]
DOCKER

# 6d.  EXECUTOR  (posts orders to Coinbase)
cat > "$PROJECT/services/executor/src/execute.py" << 'PY'
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
PY

cat > "$PROJECT/services/executor/Dockerfile" << 'DOCKER'
FROM python:3.11-slim
WORKDIR /app
COPY ../../.. /app
RUN pip install --no-cache-dir -e .
CMD ["python","-m","services.executor.src.execute"]
DOCKER

# 6e.  PUBLIC API  (FastAPI)
cat > "$PROJECT/services/api/src/main.py" << 'PY'
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
PY

cat > "$PROJECT/services/api/Dockerfile" << 'DOCKER'
FROM python:3.11-slim
WORKDIR /app
COPY ../../.. /app
RUN pip install --no-cache-dir -e .
CMD ["uvicorn","services.api.src.main:app","--host","0.0.0.0","--port","8000"]
DOCKER

# 7.  docker-compose.yml
cat > "$PROJECT/infra/docker-compose.yml" << 'YML'
version: "3.9"
services:
  kafka:
    image: vectorized/redpanda
    command: redpanda start --overprovisioned
    ports: ["9092:9092"]

  qdrant:
    image: qdrant/qdrant:v1.8
    ports: ["6333:6333"]

  ingest:
    build: ../services/ingest
    env_file: ../.env.example
    depends_on: [kafka]

  macro_agent:
    build: ../services/macro_agent
    env_file: ../.env.example
    depends_on: [kafka,qdrant]

  strategy_agent:
    build: ../services/strategy_agent
    env_file: ../.env.example
    depends_on: [kafka,qdrant]

  executor:
    build: ../services/executor
    env_file: ../.env.example
    depends_on: [kafka]

  api:
    build: ../services/api
    ports: ["8000:8000"]
    env_file: ../.env.example
    depends_on: [kafka]
YML

# 8.  Pre-commit + helper script
cat > "$PROJECT/.pre-commit-config.yaml" << 'YAML'
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.4.4
  hooks: [ {id: ruff} ]
YAML

cat > "$PROJECT/scripts/bootstrap.sh" << 'BASH'
#!/usr/bin/env bash
poetry install
cp .env.example .env.local 2>/dev/null || true
echo "🚀  Project ready. Edit .env.local then run:  docker-compose -f infra/docker-compose.yml up --build"
BASH
chmod +x "$PROJECT/scripts/bootstrap.sh"

echo "✅  $PROJECT skeleton generated."
