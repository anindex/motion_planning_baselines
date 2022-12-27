from abc import ABC, abstractmethod
from typing import Tuple
import torch


class MPPlanner(ABC):
    """Base class for all planners."""

    def __init__(self, name: str, tensor_args: dict = None, **kwargs):
        self.name = name
        if tensor_args is None:
            tensor_args = {
                'device': torch.device('cpu'),
                'dtype': torch.float32,
            }
        self.tensor_args = tensor_args
        self._kwargs = kwargs

    @abstractmethod
    def optimize(self, opt_iters: int = 1, **observation) -> Tuple[bool, torch.Tensor]:
        """Plan a path from start to goal.

        Args:
            opt_iters: Number of optimization iters.
            observation: dict of observations.

        Returns:
            success: True if a path was found.
            path: Path from start to goal.
        """
        pass

    def __call__(self, opt_iters: int = 1, **observation) -> Tuple[bool, torch.Tensor]:
        """Plan a path from start to goal.

        Args:
            start: Start position.
            goal: Goal position.

        Returns:
            success: True if a path was found.
            path: Path from start to goal.
        """
        return self.optimize(opt_iters, **observation)

    def __repr__(self):
        return f"{self.name}({self._kwargs})"