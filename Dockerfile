FROM python:3.12-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone the LLM repository
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/hung20gg/llm.git
RUN pip install -r llm/quick_requirements.txt

# Copy the rest of the application code (excluding .env file)
COPY . .

# Set Python path to include the current directory
ENV PYTHONPATH=/app:${PYTHONPATH}

# Default command to run tests
CMD ["python", "-m", "test.test"]