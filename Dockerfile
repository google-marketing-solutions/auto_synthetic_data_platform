FROM python:3.10

RUN pip install --require-hashes  --upgrade pip

WORKDIR /auto_synthetic_data_platform

COPY . . 

RUN pip install --require-hashes -r requirements.txt