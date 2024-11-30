FROM python:3.12.7-slim

WORKDIR /fastapp

ADD configs/fastapp/requirements.txt /fastapp/
RUN pip install --no-cache-dir -r requirements.txt

ADD fastapp.py /fastapp/
ADD src/ /fastapp/src/
ADD models/ /fastapp/models/
ADD data/processed/preprocessor.pkl /fastapp/data/processed/preprocessor.pkl

CMD [ "uvicorn", "fastapp:app", "--port", "8000", "--host", "0.0.0.0" ]


