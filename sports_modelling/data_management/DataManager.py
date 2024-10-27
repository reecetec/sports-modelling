from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Optional, Union, Dict
import logging
import os


class DataManager:
    """Base class for handling data operations across different sports."""
    
    def __init__(self, sport_subdir: Optional[str] = None):
        """
        Initialize DataManager with optional sport subdirectory.
        
        Args:
            sport_subdir (str, optional): Subdirectory name for specific sport data
        """
        self.root_dir = self._find_project_root(Path(os.getcwd()))
        self.data_dir = self.root_dir / 'data'
        
        if sport_subdir:
            self.data_dir = self.data_dir / sport_subdir
            
        self._setup_logging()
        self._ensure_directories()
    
    @staticmethod
    def _find_project_root(current_dir: Path) -> Path:
        """
        Find the project root directory by looking for pyproject.toml
        
        Args:
            current_dir (Path): Starting directory for search
            
        Returns:
            Path: Project root directory
        """
        while current_dir != current_dir.parent:
            if (current_dir / 'pyproject.toml').exists():
                return current_dir
            current_dir = current_dir.parent
        
        # If we can't find the root, use the current directory
        return Path(os.getcwd())
    
    def _setup_logging(self):
        """Configure logging for the data manager."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save_parquet(self, df: pd.DataFrame, filename: str, partition_cols: Optional[list] = None):
        """
        Save DataFrame to parquet with optional partitioning.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Relative path and filename within data directory
            partition_cols (list, optional): Columns to partition by
        """
        filepath = self.data_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if partition_cols:
                df.to_parquet(filepath, partition_cols=partition_cols)
            else:
                df.to_parquet(filepath)
            self.logger.info(f"Successfully saved data to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {str(e)}")
            raise
    
    def load_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from parquet file.
        
        Args:
            filename (str): Relative path and filename within data directory
            
        Returns:
            pd.DataFrame: Loaded data
        """
        filepath = self.data_dir / filename
        try:
            df = pd.read_parquet(filepath)
            self.logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise