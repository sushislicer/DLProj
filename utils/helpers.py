"""
Utility helper functions for the Qwen quantization pipeline.
"""

import os
import torch
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: str = "pipeline.log",
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Create a configured logger (file + console).

    This project uses multiple entrypoints (pipeline, benchmarking, reporting).
    `logging.basicConfig()` is global and becomes a no-op after first use, so we
    attach handlers directly to a named logger and avoid duplicate handlers.

    Args:
        log_dir: Directory to save log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Log filename within `log_dir`.
        logger_name: Logger name. If omitted, uses this module's `__name__`.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    name = logger_name or __name__
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent double logging via root handlers.
    logger.propagate = False

    # Avoid adding duplicate handlers if setup_logging() is called multiple times.
    desired_log_path = os.path.join(log_dir, log_file)
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(desired_log_path)
        for h in logger.handlers
    )
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not has_file_handler:
        fh = logging.FileHandler(desired_log_path)
        fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not has_stream_handler:
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_size(model: torch.nn.Module, model_name: str = "Model") -> None:
    """
    Print model size information.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\n{model_name} Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%\n")


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that configuration contains all required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProgressTracker:
    """Track progress of multi-stage pipeline."""
    
    def __init__(self, stages: list, log_dir: str = "logs"):
        """
        Initialize progress tracker.
        
        Args:
            stages: List of stage names
            log_dir: Directory to save progress file
        """
        self.stages = stages
        self.current_stage = 0
        self.log_dir = log_dir
        self.progress_file = os.path.join(log_dir, "progress.txt")
        ensure_dir(log_dir)
        self._load_progress()
    
    def _load_progress(self) -> None:
        """Load progress from file if exists."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                stage_name = f.read().strip()
                if stage_name in self.stages:
                    self.current_stage = self.stages.index(stage_name)
    
    def save_progress(self) -> None:
        """Save current progress to file."""
        with open(self.progress_file, 'w') as f:
            f.write(self.stages[self.current_stage])
    
    def advance_stage(self) -> None:
        """Advance to the next stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.save_progress()
    
    def get_current_stage(self) -> str:
        """Get current stage name."""
        return self.stages[self.current_stage]
    
    def is_complete(self) -> bool:
        """Check if all stages are complete."""
        return self.current_stage >= len(self.stages) - 1
    
    def reset(self) -> None:
        """Reset progress to beginning."""
        self.current_stage = 0
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
