FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt app/requirements.txt
RUN pip install --upgrade pip 
RUN pip install -r app/requirements.txt

COPY . /app

ENV PYTHONPATH="/app"

ENTRYPOINT ["python", "src/entrypoint.py"]
