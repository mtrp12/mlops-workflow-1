FROM python:3.12.7-slim

WORKDIR /locust

ADD configs/locust/requirements.txt /locust/
RUN pip install --no-cache-dir -r requirements.txt
ADD locustfile.py /locust/

CMD [ "locust", "--web-port", "8089" ]
