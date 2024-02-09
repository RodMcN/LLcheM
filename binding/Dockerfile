FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN mkdir LLcheM
WORKDIR /workdir
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . ./
ENTRYPOINT ["python", "train_binding_x.py"]