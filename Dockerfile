FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN mkdir LLcheM
WORKDIR /LLcheM
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py .
ENTRYPOINT ["python", "run_training.py"]