FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN pip install --upgrade pip

COPY . /app

RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="/app"

ENTRYPOINT ["python", "src/entrypoint.py"]
