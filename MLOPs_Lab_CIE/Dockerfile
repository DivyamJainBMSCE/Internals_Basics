FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir pandas numpy scikit-learn

CMD ["python", "src/predict_cli.py"]