import os
import mlflow
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .base_logger import BaseLogger
import logging

class MLflowLogger(BaseLogger):
    """MLflow implementation of BaseLogger."""
    
    def __init__(self):

        super().__init__()

        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Default")
        self.run_id = None
        self.experiment_id = None
        self.tracking_backend = "mlflow"
        
        
    def login(self, **kwargs):
        """Set up MLflow tracking."""
        tracking_uri = kwargs.get('tracking_uri', self.tracking_uri)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
    def init_run(self, project: str = None, entity: str = None, job_type: str = "experiment",
                 config: Dict[str, Any] = {}, name: Optional[str] = None, train_id = None) -> Any:
        
        if self.run is not None:
            logging.warning("Run already initialized. Please finish the current run before starting a new one.")
            return self.run
        
        """Initialize a new MLflow run."""
        # Use project as experiment name if provided
        experiment_name = project or self.experiment_name
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        self.experiment_id = experiment_id

        # Start the run
        tags = {"job_type": job_type}
        if entity:
            tags["entity"] = entity

        self.config = config
            
        mlflow.end_run()

        active_run = mlflow.start_run(
            run_name=name,
            experiment_id=experiment_id,
            tags=tags
        )

        self.run = mlflow.tracking.MlflowClient()
        self.run_id = active_run.info.run_id

        if train_id:
            if self.run.get_run(train_id):
                logging.info(f"Connecting to train run with ID: {train_id}")
                self.train_id = train_id
            else:
                logging.warning(f"Train run with ID: {train_id} not found")

        else:
            logging.info(f"Starting isolate run with ID: {name}")
        
        # Log the config as parameters
        if config:
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                    
        return self.run
    
    def log_metric(self, key: str, value: Union[float, int], run_id = None, **kwargs) -> None:
        """Log a single metric to MLflow."""
        if not self.check_run_status():
            return
        
        run_id  = self.get_run_id(run_id)
        
        if run_id:
            self.run.log_metric(run_id = run_id, key = key, value=value, **kwargs)
        
    def log_metrics(self, metrics: Dict[str, Union[float, int]], run_id = None) -> None:
        """Log multiple metrics to MLflow."""
        if not self.check_run_status():
            return
        
        step = metrics.pop('current_steps', 0)
        for key, value in metrics.items():
            self.log_metric(key, value, step=step, run_id=run_id)   
    
    def log_table(self, key: str, dataframe: pd.DataFrame, run_id: str = None) -> None:
        """Log a dataframe as a table to MLflow."""
        if not self.check_run_status():
            return
        
        if not key.endswith(".json"):
            key = f"{key}.json"

        run_id  = self.get_run_id(run_id)
        if run_id:
            self.run.log_table(run_id = run_id, data = dataframe, artifact_file = key)

        # # Save dataframe to CSV and log it
        # temp_path = f"/tmp/{key}.csv"
        # dataframe.to_csv(temp_path, index=False)
        # self.log_artifact(temp_path, f"tables/{key}", run_id=run_id)
        
        # # Clean up
        # try:
        #     os.remove(temp_path)
        # except:
        #     pass
    
    def log_artifact(self, local_path: str, name: Optional[str] = None, type_ = "file", run_id: str = None) -> str:
        """Log an artifact file to MLflow and return the artifact path.
        
        Args:
            local_path: Path to the local file to log
            name: Optional artifact path to log to within the run
            type_: Type of artifact (default: "file")
            
        Returns:
            String path to the logged artifact in MLflow
        """
        if not self.check_run_status():
            return None
        
        run_id  = self.get_run_id(run_id)

        artifact_path = name or ""
        
        if run_id:
            self.run.log_artifact(run_id = run_id, local_path = local_path, artifact_path = artifact_path)
        
            # Construct the artifact path in MLflow
            file_name = os.path.basename(local_path)
            if artifact_path:
                full_artifact_path = f"{artifact_path}/{file_name}"
            else:
                full_artifact_path = file_name
                
            return full_artifact_path
        
        else:
            logging.warning("Run ID is not available. Cannot log artifact.")
            return None

    def log_artifacts(self, local_dir: str, name: Optional[str] = None, type_ = "file", run_id: str = None) -> str:
        """Log an artifact file to MLflow and return the artifact path.
        
        Args:
            local_path: Path to the local file to log
            name: Optional artifact path to log to within the run
            type_: Type of artifact (default: "file")
            
        Returns:
            String path to the logged artifact in MLflow
        """
        if not self.check_run_status():
            return None
        
        run_id  = self.get_run_id(run_id)
        
        if run_id:
            artifact_path = name or ""
            self.run.log_artifacts(run_id = run_id, local_dir = local_dir, artifact_path = artifact_path)
            
            # Construct the artifact path in MLflow
            file_name = os.path.basename(local_dir)
            if artifact_path:
                full_artifact_path = f"{artifact_path}/{file_name}"
            else:
                full_artifact_path = file_name
                
            return full_artifact_path
        else:
            logging.warning("Run ID is not available. Cannot log artifact.")
            return None
    

    def update_summary(self, key: str, value: Any, run_id = None) -> None:
        """Update a summary metric in MLflow."""
        # MLflow doesn't have a direct equivalent to wandb's summary
        # We can use log_metric instead
        self.log_metric(f"summary_{key}", value, run_id=run_id)
    
    def finish_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self.run_id = None
        self.run = None

    def get_tracking_url(self) -> Optional[str]:
        """Get URL to the current run in MLflow UI."""
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri and self.run_id:
            # For hosted MLflow or local server
            if tracking_uri.startswith('http'):
                return f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
        return None


    def get_model_checkpoint_step(self, model_name: Optional[str] = None) -> int:
        
        if not self.checkpoint_step:
            model_name_to_use = model_name or getattr(self, 'model_name', None)
            if not model_name_to_use:
                logging.warning("No model name provided for evaluation status update.")
                self.checkpoint_step = 0
                return 0
            
            # get model checkpoint based on model tag
            model_metadata = self.run.get_model_version(model_name_to_use, self.version)
            model_checkpoint_step = model_metadata.tags.get('checkpoint', None)
            if model_checkpoint_step and str(model_checkpoint_step).isdigit():
                self.checkpoint_step = int(model_checkpoint_step)
                logging.info(f"Model checkpoint step: {self.checkpoint_step}")
                return int(model_checkpoint_step)
            
            logging.warning(f"Model checkpoint step not found for model name: {model_name_to_use}")
            self.checkpoint_step = 0
            return 0
        else:
            return self.checkpoint_step
    
    
    def update_evaluation_status(self, status: str, result: Optional[Dict] = None, model_name: Optional[str] = None) -> None:
        """Update the evaluation status in MLflow."""
        if not self.check_run_status():
            return
        
        model_name_to_use = model_name or getattr(self, 'model_name', None)
        if not model_name_to_use:
            logging.warning("No model name provided for evaluation status update.")
            return
        
        self.run.set_model_version_tag(
            name = model_name_to_use,
            version = self.version,
            key = "evaluate",
            value = status
        )

        self.run.set_model_version_tag(
            name = model_name_to_use,
            version = self.version,
            key = "evaluation_id",
            value = self.run_id
        )
        
        
        checkpoint_step = self.get_model_checkpoint_step(model_name_to_use)

        if result:
            dataset_name, evaluation_result = zip(*result.items())
            dataset_name = dataset_name[0]
            evaluation_result = evaluation_result[0]

            self.run.set_model_version_tag(
                name = model_name_to_use,
                version = self.version,
                key = "mean_score",
                value = evaluation_result
            )

            logging.info(f"Evaluation result for {dataset_name}: {evaluation_result}")

            # Evaluation run
            self.log_metric(key = dataset_name, value = evaluation_result)
            
            # Training run
            if self.train_id:
                self.log_metric(
                    key = dataset_name,
                    value = evaluation_result,
                    run_id = self.train_id,
                    step = checkpoint_step
                )
        else:
            dataset_name = None

        # Update champion/challenger status
        if status == 'completed' and dataset_name:
            filter_name = f"name='{model_name_to_use}'"

            model_versions = self.run.search_model_versions(filter_name)

            if model_versions:
                selected_version = []
                for mv in model_versions:
                    # if mv.tags.get('evaluate') == 'completed':
                        
                    mv_run_id = mv.tags.get('evaluation_id', None)

                    if mv_run_id:
                        mv_evaluation_result = self.run.get_run(mv_run_id).data.metrics.get(dataset_name, None)

                        if mv_evaluation_result:
                        
                            selected_version.append({
                                'model_name': mv.name,
                                'version': mv.version,
                                'run_id': mv.run_id,
                                'evaluation_result': mv_evaluation_result
                            })

                if selected_version:
                    # Rank to get the best and second best
                    selected_version = sorted(selected_version, key=lambda x: x['evaluation_result'], reverse=True)
                    champion_version = selected_version[0]
                    challenger_version = selected_version[1] if len(selected_version) > 1 else None

                    logging.info(f"Champion version: {champion_version['version']} with result: {champion_version['evaluation_result']}")
                    logging.info(f"Challenger version: {challenger_version['version']} with result: {challenger_version['evaluation_result']}" if challenger_version else "No challenger version found.")
                    # Update the champion/challenger alias

                    self.run.set_registered_model_alias(
                        name = model_name_to_use,
                        version = champion_version['version'],
                        alias = "champion"
                    )

                    if challenger_version:
                        self.run.set_registered_model_alias(
                            name = model_name_to_use,
                            version = challenger_version['version'],
                            alias = "challenger"
                        )
                    else:
                        logging.warning("No challenger version found.")




                else:
                    logging.warning(f"No completed model versions found for model name: {model_name_to_use}")
            
            else:
                logging.warning(f"No model versions found for model name: {model_name_to_use}")

                # Get the results of the selected version

        
        
        # pass
        # try:
        #     self.run.search_runs(

        #     )