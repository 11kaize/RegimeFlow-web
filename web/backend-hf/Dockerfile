FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN pip install --no-cache-dir --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# HF Spaces expects port 7860
ENV PORT=7860

CMD ["python", "app.py"]
