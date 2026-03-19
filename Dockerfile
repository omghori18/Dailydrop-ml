FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir flask gunicorn pandas prophet firebase-admin

COPY . .

EXPOSE 8080

CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8080", "app:app"]