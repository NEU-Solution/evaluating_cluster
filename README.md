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


Run docker image
```bash
# Build the Docker image
docker build -t evaluate_cluster .

# Run the container with tests
docker run evaluate_cluste
```

Run via docker compose