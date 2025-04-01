# Evaluating Cluster

This will be dockerized to push it into wandb, and wandb will invoke it as queue for evaluation

- [Youtube](https://www.youtube.com/watch?v=d_TN8fIDSB8&list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&index=17)
- [Github](https://github.com/wandb/edu/tree/main/model-management)


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

Run docker image
```bash
# Build the Docker image
docker build -t evaluate_cluster .

docker build --no-cache -t evaluate_cluster .

# Run the container with tests
docker run --gpus all --env-file .env evaluate_cluster
```

Run via docker compose
```bash
docker-compose up --build -d
```