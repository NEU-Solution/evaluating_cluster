from abc import ABC, abstractmethod
import os
import sys
import logging

# Add parent directory to path so we can import modules
sys.path.append('..')
from src.load_model import download_model_regristry

logger = logging.getLogger(__name__)

class ModelRegistry(ABC):
    """Abstract base class for model registry implementations"""
    
    @abstractmethod
    def download_model(self, model_name: str, version: str = None, download_dir: str = 'models') -> str:
        """Download a model from the registry and return the path"""
        pass

class WandbModelRegistry(ModelRegistry):
    """Weights & Biases model registry implementation"""
    
    def download_model(self, model_name: str, version: str = None, download_dir: str = 'models') -> str:
        """Download a model from W&B registry"""
        logger.info(f"Downloading model {model_name}:{version or 'latest'} from W&B")
        return download_model_regristry(model_name, version, download_dir)

# Factory function to get the appropriate registry
def get_model_registry(registry_type: str = "wandb") -> ModelRegistry:
    """Get the appropriate model registry implementation"""
    if registry_type.lower() == "wandb":
        return WandbModelRegistry()
    # Add more registry types as needed
    else:
        raise ValueError(f"Unsupported registry type: {registry_type}")