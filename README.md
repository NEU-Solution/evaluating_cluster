# Evaluating Cluster

## Architecture Diagrams
![Architecture Diagram](./assets/diagrams.png)


Create code to evaluate

The evaluation dataset must be in the following format

```json
[
    {
        "id":1,
        "question": "1 + 1 = ?",
        "choices" : [
            "A. 1",
            "B. 2",
            "C. 3",
            "D. 4"
        ],
        "answer": "D"
    }
]
```

Test
```bash
python -m test.test
```

Remenber add env file
```bash
WANDB_API_KEY=your_api_key
WANDB_PROJECT=mlops
WANDB_ENTITY=neu-solution
```

## Run docker evaluation 
```bash
# Build the Docker image

# evaluation function
docker build -t evaluate_model -f Dockerfile.eval .

# evaluation server
docker build -t evaluate_model -f Dockerfile .

# Run the container with tests
docker run --gpus all --env-file .env -v ~/.cache/huggingface:/root/.cache/huggingface  evaluate_model 
```

## Run docker api service
```bash
docker build -t evaluation-api .

docker run --gpus all --env-file .env -p 23477:23477 -v ~/.cache/huggingface:/root/.cache/huggingface evaluation-api
```

## Run via docker compose
```bash
docker-compose up --build -d
```

