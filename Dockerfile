FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add Elasticsearch Python client
RUN pip install --no-cache-dir elasticsearch==8.11.0

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 9000

# Command to run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "9000", "--reload"] 