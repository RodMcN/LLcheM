FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN mkdir LLcheM
WORKDIR /LLcheM
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py .
ENTRYPOINT ["python", "run_training.py"]