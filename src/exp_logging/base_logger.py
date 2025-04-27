from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging
import threading

class BaseLogger(ABC):
    """Abstract base class for experiment tracking loggers."""
    
    _instances = {}
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super(BaseLogger, cls).__new__(cls)
                cls._instances[cls] = instance
            return cls._instances[cls]
    
    @abstractmethod
    def __init__(self):
        self.checkpoint_step = None
        self.train_id = None
        self.run = None
        self.config = {}
        pass

    def init_run(self, project: str, entity: str, job_type: str, config: Dict[str, Any] = None, 
                 name: Optional[str] = None, train_id = None) -> Any:
        """Initialize a new run."""
        pass
    
    @abstractmethod
    def log_metric(self, key: str, value: Union[float, int], run_id: str = None, **kwargs) -> None:
        """Log a single metric."""
        pass
        
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]], run_id: str = None, **kwargs) -> None:
        """Log multiple metrics."""
        pass
    
    @abstractmethod
    def log_table(self, key: str, dataframe: pd.DataFrame, run_id: str = None) -> None:
        """Log a dataframe as a table."""
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, name: Optional[str] = None, run_id: str = None, **kwargs) -> None:
        """Log an artifact file."""
        pass
        
    @abstractmethod
    def update_summary(self, key: str, value: Any) -> None:
        """Update a summary metric."""
        pass
    
    @abstractmethod
    def finish_run(self) -> None:
        """End the current run."""
        pass
    
    @abstractmethod
    def login(self, **kwargs) -> None:
        """Login to the tracking service."""
        pass

    @abstractmethod
    def get_model_checkpoint_step(self):
        """Get the model checkpoint step"""
        pass

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in the tracking UI, if available."""
        return None
    
    def auto_init_run(self) -> None:
        """Automatically initialize a run if not already done."""
        pass

    def check_run_status(self) -> bool:
        """
        Check if the current run is active.
        
        Returns:
            True if the run is active, False otherwise
        """
        flag = hasattr(self, 'run') and self.run is not None
        if not flag:
            logging.warning("No active run detected.")
        return flag
    
    def set_model_version(self, model: str, version: str) -> None:
        """Set the original version of the model."""
        self.model_name = model
        self.version = version

    def get_run_id(self, run_id = None) -> str:
        """Get the run ID."""
        if run_id is None:
            return self.run_id
    
        if run_id == 'train_id' and hasattr(self, 'train_id'):
            return self.train_id
        return run_id

    @abstractmethod
    def update_evaluation_status(self, status: str, result: Dict, model_name: Optional[str] = None) -> None:
        """Update the evaluation status."""
        pass

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the configuration."""
        if self.run is not None:
            self.config.update(config)
            # self.run.config.update(config)
        else:
            logging.warning("No active run to update config.")