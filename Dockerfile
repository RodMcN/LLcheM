FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y wget

ARG CONDA_VERSION=py310_23.3.1-0

ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -b && \
    rm -f Miniconda3-${CONDA_VERSION}-x86_64.sh

RUN mkdir LLcheM
WORKDIR /LLcheM
RUN pip3 install torch>=2 torchtext --index-url https://download.pytorch.org/whl/cu117
COPY requirements.txt .
# RUN conda install -y --file requirements.txt -c pytorch -c nvidia -c conda-forge
RUN pip install -r requirements.txt

COPY *.py .
ENTRYPOINT ["python", "run_training.py"]