FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_model.py .
EXPOSE 8080
CMD ["uvicorn", "train_model:app", "--host", "0.0.0.0", "--port", "8080"]