from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging

class BaseLogger(ABC):
    """Abstract base class for experiment tracking loggers."""
    
    @abstractmethod
    def init_run(self, project: str, entity: str, job_type: str, config: Dict[str, Any] = None, 
                 name: Optional[str] = None) -> Any:
        """Initialize a new run."""
        pass
    
    @abstractmethod
    def log_metric(self, key: str, value: Union[float, int]) -> None:
        """Log a single metric."""
        pass
        
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log multiple metrics."""
        pass
    
    @abstractmethod
    def log_table(self, key: str, dataframe: pd.DataFrame) -> None:
        """Log a dataframe as a table."""
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
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