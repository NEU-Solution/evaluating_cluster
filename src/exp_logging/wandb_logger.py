import os
import wandb
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .base_logger import BaseLogger
import logging

class WandbLogger(BaseLogger):
    """Weights & Biases implementation of BaseLogger."""
    
    def __init__(self):

        super().__init__()

        self._api_key = os.getenv("WANDB_API_KEY")
        self.project = os.getenv("WANDB_PROJECT")
        self.entity = os.getenv("WANDB_ENTITY")
        self.tracking_backend = "wandb"
        wandb.login(key=self._api_key)
        
        
    def init_run(self, project: str = None, entity: str = None, job_type: str = "experiment", 
                 config: Dict[str, Any] = {}, name: Optional[str] = None, train_id = None) -> Any:
        """Initialize a new WandB run."""
        
        if self.run is not None:
            logging.warning("Run already initialized. Please finish the current run before starting a new one.")
            return self.run

        self.config = config
        
        if train_id:
            logging.info(f"Connecting to train run with ID: {train_id}")
            self.train_id = train_id
        else:
            logging.info(f"Starting isolate run with ID: {name}")
            self.train_id = None

        self.run = wandb.init(
            project=project or self.project,
            entity=entity or self.entity,
            job_type=job_type,
            config=config,
            name=name
        )
        return self.run
    
    def log_metric(self, key: str, value: Union[float, int], run_id = None) -> None:
        """Log a single metric to WandB."""
        if not self.check_run_status():
            return
        
        self.run.log({key: value})
        
    def log_metrics(self, metrics: Dict[str, Union[float, int]], run_id = None) -> None:
        """Log multiple metrics to WandB."""
        if not self.check_run_status():
            return
        
        self.run.log(metrics)
    
    def log_table(self, key: str, dataframe: pd.DataFrame, run_id = None) -> None:
        """Log a dataframe as a table to WandB."""
        if not self.check_run_status():
            return
        
        table = wandb.Table(dataframe=dataframe)
        self.run.log({key: table})
    
    def log_artifact(self, local_path: str, name: Optional[str] = None, run_id = None) -> None:
        """Log an artifact file to WandB."""
        if not self.check_run_status():
            return
        
        artifact = wandb.Artifact(name=name or os.path.basename(local_path), type="dataset")
        artifact.add_file(local_path)
        self.run.log_artifact(artifact)
        
    def update_summary(self, key: str, value: Any) -> None:
        """Update a summary metric in WandB."""
        if not self.check_run_status():
            return
        
        self.run.summary[key] = value
    
    def finish_run(self) -> None:
        """End the current WandB run."""
        if self.run:
            self.run.finish()
            self.run = None
            self.run_id = None

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in WandB UI."""
        if self.run:
            return self.run.get_url()
        return None