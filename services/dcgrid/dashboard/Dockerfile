FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -U pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY services /app/services

ENV PYTHONPATH=/app/services

EXPOSE 8501

CMD ["streamlit", "run", "/app/services/dcgrid/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
