# Running the Evaluation API

To run the evaluation API, use the following command:

```bash
cd api
python evaluation_api.py
```

This will start the server on port 23477. Below are curl examples for interacting with the API.

## Curl Examples

### 1. Start a Basic Evaluation Job

```bash
curl -X 'POST' \
  'http://localhost:23477/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "base_model_name": "Qwen/Qwen3-1.7B",
  "lora_model_name": "models:/thinking/8",
  "data_version": "latest",
  "multi_thread": true,
  "llm_backend": "vllm",
  "max_workers": 3,
  "port": 8000,
  "tracking_backend": "mlflow",
  "train_id": "dbf02083c4ca415e9eac397b28df1e6c"
}'
```

### 2. Start Evaluation with MLflow Backend and Webhook

```bash
curl -X 'POST' \
  'http://localhost:23477/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "base_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "lora_model_name": "wandb-registry-model/initial-sft",
  "data_version": "v0.1",
  "tracking_backend": "mlflow",
  "webhook_url": "https://webhook.site/your-unique-id"
}'
```

### 3. Check Status of a Specific Evaluation Job

```bash
curl -X 'GET' \
  'http://localhost:23477/evaluate/YOUR_JOB_ID' \
  -H 'accept: application/json'
```

### 4. List All Evaluation Jobs

```bash
curl -X 'GET' \
  'http://localhost:23477/evaluate' \
  -H 'accept: application/json'
```

### 5. Check Queue Status

```bash
curl -X 'GET' \
  'http://localhost:23477/queue' \
  -H 'accept: application/json'
```

### 6. Start Evaluation with Reject Strategy (Don't Queue)

```bash
curl -X 'POST' \
  'http://localhost:23477/evaluate?strategy=reject' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "base_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "lora_model_name": "wandb-registry-model/initial-sft",
  "data_version": "latest"
}'
```

### 7. Check API Status

```bash
curl -X 'GET' \
  'http://localhost:23477/' \
  -H 'accept: application/json'
```

The API follows an asynchronous pattern, so evaluation jobs run in the background while you can check their status using the provided job ID. The webhook URL, if provided, will receive notifications when the job completes or fails.