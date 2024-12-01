from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from pydantic import BaseModel
from src.logger.basic_logging import logging
from random import randint
import redis

from src.pipelines.prediction_pipeline import CustomerChurnPredictionPipeline, CustomerData


# ==========================================
# SETUP
# ==========================================
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


pool = redis.ConnectionPool(host='redis.example.com', port=6379, db=0)
r = redis.Redis(connection_pool=pool)

prediction_pipeline = CustomerChurnPredictionPipeline()
# prediction_pipeline = None

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

# ==========================================
# Data Models
# ==========================================
class SquareResponse(BaseModel):
    input: int
    output: int
    cached: bool

class PredictModel(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

class PredictResponse(BaseModel):
    customerid: str
    churn: str
    churn_raw: int
    cached: bool

class ErrorResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    message: str

# ==========================================
# ROUTES
# ==========================================
@app.get("/health")
async def health() -> HealthResponse:
    """
    Shows application health status.
    """
    return HealthResponse("Healthy")

@app.post("/predict",responses={
        200: {"model": PredictResponse, "description": "Successful Operation"},
        400: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    })
async def predict(customer: PredictModel) -> PredictResponse|ErrorResponse:
    try:
        # Get JSON data from request
        json_data = customer.model_dump()
        logging.info(f"INCOMING_PARAMS: {json_data}")
        
        cached = True
        ret_val = r.get(customer.customerID)
        if ret_val is not None:
            return PredictResponse(customerid=customer.customerID, churn="Yes" if int(ret_val) == 1 else "No", churn_raw=int(ret_val), cached=cached)
        
        # Create CustomClass instance with JSON data
        data = CustomerData(
            customer_id=str(json_data.get('customerID')),
            gender=str(json_data.get('gender')),
            senior_citizen=int(json_data.get('SeniorCitizen')),
            partner=str(json_data.get('Partner')),
            dependents=str(json_data.get('Dependents')),
            tenure=int(json_data.get('tenure')),
            phone_service=str(json_data.get('PhoneService')),
            multiple_lines=str(json_data.get('MultipleLines')),
            internet_service=str(json_data.get('InternetService')),
            online_security=str(json_data.get('OnlineSecurity')),
            online_backup=str(json_data.get('OnlineBackup')),
            device_protection=str(json_data.get('DeviceProtection')),
            tech_support=str(json_data.get('TechSupport')),
            streaming_tv=str(json_data.get('StreamingTV')),
            streaming_movies=str(json_data.get('StreamingMovies')),
            contract=str(json_data.get('Contract')),
            paperless_billing=str(json_data.get('PaperlessBilling')),
            payment_method=str(json_data.get('PaymentMethod')),
            monthly_charges=float(json_data.get('MonthlyCharges')),
            total_charges=str(json_data.get('TotalCharges'))


        )

        # Get prediction
        final_data = data.get_pd_dataframe()
        pred = prediction_pipeline.predict(final_data)
        r.set(customer.customerID, int(pred[0]))

        # Return prediction result
        return PredictResponse(customerid=customer.customerID, churn="Yes" if pred[0] == 1 else "No", churn_raw=int(pred[0]), cached=False)

    except Exception as e:
        logging.error("Churn prediction failed", exc_info=e)
        return HTTPException(status_code=500, detail=ErrorResponse(status="error", message=str(e)).model_dump())

@app.get("/square",responses={
        200: {"model": SquareResponse, "description": "Successful Operation"},
        400: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    })
async def square(number: int) -> SquareResponse|ErrorResponse:
    """
    Accepts an integer as input and returns it's square value. 10% chance to fail.
    """
    p: int = randint(1, 100)
    # probability of failure less or equal 10
    if p > 10:
        ret_val = r.get(number)
        cached = True
        if ret_val is None:
            ret_val = number * number
            r.set(number, ret_val)
            cached = False
        return SquareResponse(input=number, output=int(ret_val), cached=cached)
    else:
        try:
            a = 1/0
        except Exception as e:
            logging.error("Fail Simulation", exc_info=e)
            raise HTTPException(status_code=500, detail=ErrorResponse(status="failed", message=str(e)).model_dump())
