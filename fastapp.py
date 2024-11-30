from fastapi import FastAPI
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

# Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {"message": "Hello, Prometheus!"}

@app.get("/health")
async def health():
    return {"message": "Healthy!"}

@app.get("/predict")
async def predict():
    p: int = randint(1, 100)
    # simulate 30% failure
    if p > 10:
        return {"predict": "ON"}
    else:
        try:
            a = 1/0
        except Exception as e:
            logging.error("Fail Simulation", exc_info=e)
            raise e
