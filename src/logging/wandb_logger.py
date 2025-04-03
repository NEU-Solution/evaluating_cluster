import os
import wandb
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .base_logger import BaseLogger

class WandbLogger(BaseLogger):
    """Weights & Biases implementation of BaseLogger."""
    
    def __init__(self):
        self.run = None
        self.api_key = os.getenv("WANDB_API_KEY")
        self.project = os.getenv("WANDB_PROJECT")
        self.entity = os.getenv("WANDB_ENTITY")
        self.tracking_backend = "wandb"
        
    def login(self, **kwargs):
        """Login to WandB."""
        key = kwargs.get('key', self.api_key)
        wandb.login(key=key)
        
    def init_run(self, project: str = None, entity: str = None, job_type: str = "experiment", 
                 config: Dict[str, Any] = None, name: Optional[str] = None) -> Any:
        """Initialize a new WandB run."""
        self.run = wandb.init(
            project=project or self.project,
            entity=entity or self.entity,
            job_type=job_type,
            config=config,
            name=name
        )
        return self.run
    
    def log_metric(self, key: str, value: Union[float, int]) -> None:
        """Log a single metric to WandB."""
        self.run.log({key: value})
        
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log multiple metrics to WandB."""
        self.run.log(metrics)
    
    def log_table(self, key: str, dataframe: pd.DataFrame) -> None:
        """Log a dataframe as a table to WandB."""
        table = wandb.Table(dataframe=dataframe)
        self.run.log({key: table})
    
    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        """Log an artifact file to WandB."""
        artifact = wandb.Artifact(name=name or os.path.basename(local_path), type="dataset")
        artifact.add_file(local_path)
        self.run.log_artifact(artifact)
        
    def update_summary(self, key: str, value: Any) -> None:
        """Update a summary metric in WandB."""
        self.run.summary[key] = value
    
    def finish_run(self) -> None:
        """End the current WandB run."""
        if self.run:
            self.run.finish()
            self.run = None

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in WandB UI."""
        if self.run:
            return self.run.get_url()
        return None