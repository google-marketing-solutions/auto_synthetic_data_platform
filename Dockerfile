FROM python:3.10

RUN pip install --upgrade pip

WORKDIR /auto_synthetic_data_platform

COPY . . 

RUN pip install -r requirements.txt