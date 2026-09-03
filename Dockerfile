FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DATABASE_URL is provided at runtime (docker run -e DATABASE_URL=... or via --env-file)
CMD ["python", "main.py"]
