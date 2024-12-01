from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from src.logger.basic_logging import logging
from random import randint

app = FastAPI()

# Automatic instrumentation and metrics exposure
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True
    ).add(
        metrics.default()
    ).add(
        metrics.latency()
    ).add(
        metrics.requests()
    ).instrument(app).expose(app)

import redis

pool = redis.ConnectionPool(host='redis.example.com', port=6379, db=0)
r = redis.Redis(connection_pool=pool)

BASE_PATH: str = '/fastapp'
BASE_PATH_LEN: int = len(BASE_PATH)
# expose api under /fastapp subpath for reverse proxy
@app.middleware("http")
async def base_path_middleware(request: Request, call_next):
    if request.url.path.startswith(BASE_PATH):
        # Remove the base path before processing
        request.scope["path"] = request.url.path[BASE_PATH_LEN :]
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {"message": "Hello, Prometheus!"}

@app.get("/health")
async def health():
    return {"message": "Healthy!"}

@app.get("/predict")
async def predict(number: int):
    p: int = randint(1, 100)
    # probability of failure less or equal 10
    if p > 10:
        ret_val = r.get(number)
        is_cached = True
        if ret_val is None:
            ret_val = number * number
            r.set(number, ret_val)
            is_cached = False
        return {"input": number, "output": int(ret_val), "is_cached": is_cached}
    else:
        try:
            a = 1/0
        except Exception as e:
            logging.error("Fail Simulation", exc_info=e)
            raise e
