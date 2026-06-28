"""
Classes for generating statistics
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

class FailReason(str, Enum):
    TIMEOUT = "timeout"
    STUCK = "robot stuck"

@dataclass
class EpisodeStats:
    """Collects statistics about a single episode"""

    task_name: str
    """Name of the resolved task"""

    variation_id: int
    """Variation number"""

    episode_number: int
    """Episode number"""

    language_instruction: Union[str, List[str]]
    """Language instruction used to resolve the task"""

    steps: Optional[int]
    """The number of steps in this episode"""

    success: bool
    """If this episode succeded"""

    fail_reason: Optional[str]

    video_path: Optional[Union[str, Path]]
    """Path to video file from root results folder"""

    gif_path: Optional[Union[str, Path]]
    """Path to gif file from root results folder"""

    experiment_id: Any
    """Identifier for the experiment"""

    inference_time: Optional[int]
    """How many seconds did the episode take for running"""

    inference_mean_time: Optional[int]
    """Mean time for each inference step"""

@dataclass
class VariationStats:
    """Collects statistics about a set of episode for the specific variation"""
    
    task_name: str
    """Name of the resolved task"""

    variation_id: int
    """Variation number"""

    experiment_id: Any
    """Identifier for the experiment"""
    
    episodes_stats: List[EpisodeStats]
    """Stats for each episode"""

    total_episodes: int
    """The number of total episodes"""

    total_success: int
    """The number of total success episodes"""

    total_failed: int
    """The number of total failed episodes"""

    success_rate: float
    """Rate of variation success"""

@dataclass
class TaskStats:
    """Collects statistics about a set of variation"""

    task_name: str
    """Name of resolved task"""

    total_variations: int
    """Number of total variations"""

    total_episodes: int
    """Number of total episodes"""

    total_success: int
    """Number of success episodes"""

    total_failed: int
    """Number of failed episodes"""

    total_success_rate: float
    """Rate of total episode success"""

    experiment_id: Any
    """Identifier for the experiment"""

    variation_success_rates: List[float]
    """List of variations success rates"""

    variation_stats: List[VariationStats]
    """List of variation stats"""

@dataclass
class DatasetStats:
    """Collects statistics about entire dataset of tasks"""

    dataset_name: str
    """Name of evaluated dataset"""

    experiment_id: Any
    """Identifier for the experiment"""

    task_names: List[str]
    """List of all tasks"""

    total_episodes: int
    """Number of total episodes"""

    total_success: int
    """Number of success episodes"""

    total_failed: int
    """Number of failed episodes"""

    total_success_rate: float
    """Rate of total episode success"""

    tasks_succes_rates: List[float]
    """List of tasks success rates"""

    tasks_stats: List[TaskStats]
    """List of task stats"""


    