from typing import List

from torch.optim import Optimizer
from torch.nn import Module

from piepline.train_config.stages import AbstractStage

try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm


__all__ = ['TrainConfig']


class TrainConfig:
    """
    Train process setting storage

    :param train_stages: list of stages for train loop
    :param loss: loss criterion
    :param optimizer: optimizer object
    """

    def __init__(self, model: Module, train_stages: [], loss: Module, optimizer: Optimizer):
        self._train_stages = train_stages
        self._loss = loss
        self._optimizer = optimizer
        self._model = model

    def loss(self) -> Module:
        """
        Get loss object

        :return: loss object
        """
        return self._loss

    def optimizer(self) -> Optimizer:
        """
        Get optimizer object

        :return: optimizer object
        """
        return self._optimizer

    def stages(self) -> List[AbstractStage]:
        """
        Get list of stages

        :return: list of stages
        """
        return self._train_stages

    def model(self) -> Module:
        return self._model
