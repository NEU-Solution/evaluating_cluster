# testing_server.py
from fastapi import FastAPI, HTTPException
import requests
import random

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# A sample evaluation endpoint which receives the checkpoint info.
@app.post("/evaluate")
async def evaluate(checkpoint: str):
    # Simulate loading checkpoint and evaluation logic.
    # In practice, load model using the checkpoint and run evaluation.
    evaluation_score = random.uniform(0, 1)  # Dummy evaluation score.
    evaluation_results = {
        "checkpoint": checkpoint,
        "evaluation_score": evaluation_score,
        "passed_threshold": evaluation_score > 0.7  # Example threshold.
    }
    
    # Optionally, log evaluation results to Prometheus/WandB.
    
    # If performance meets criteria, trigger deployment.
    if evaluation_results["passed_threshold"]:
        try:
            deploy_response = requests.post(
                "http://deployment-server:8002/deploy",
                json={"checkpoint": checkpoint}
            )
            deploy_response.raise_for_status()
            evaluation_results["deployment_status"] = deploy_response.json()
        except Exception as e:
            evaluation_results["deployment_status"] = f"Deployment failed: {e}"
    
    return evaluation_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
