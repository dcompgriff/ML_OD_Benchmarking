FROM python:3.6.4-jessie


RUN apt-get update && \
    apt-get install git \

RUN git clone https://github.com/dcompgriff/ML_Deepnet_Object_Detection_Benchmarking.git




