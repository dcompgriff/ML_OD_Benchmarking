FROM python:3.6.4-jessie


RUN apt-get update -y \
    && apt-get install git -y

RUN git clone https://github.com/dcompgriff/ML_OD_Benchmarking.git
WORKDIR "/ML_OD_Benchmarking"

# CMD python hello_condor_docker.py



