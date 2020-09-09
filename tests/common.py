import os
import shutil
import unittest

from torch import Tensor
from torch.nn import functional as F
import numpy as np

from piepline.train_config.metrics import AbstractMetric

__all__ = ['UseFileStructure', 'data_remove', 'SimpleMetric']


class UseFileStructure(unittest.TestCase):
    base_dir = 'data'
    monitors_dir = 'monitors'
    checkpoints_dir = 'checkpoints_dir'

    def tearDown(self):
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)


def data_remove(func: callable) -> callable:
    def res(*args, **kwargs):
        ret = func(*args, **kwargs)
        UseFileStructure().tearDown()
        return ret

    return res


class SimpleMetric(AbstractMetric):
    def __init__(self, name: str = None, coeff: float = 1, collect_values: bool = False):
        super().__init__('SimpleMetric' if name is None else name)
        self._coeff = coeff
        self._collect_values = collect_values
        self._inputs = []

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        if self._collect_values:
            if len(self._values) == 0:
                self._inputs = []
            self._inputs.append((output.clone(), target.clone()))
        res = F.pairwise_distance(output, target, p=2).cpu().detach().numpy() * self._coeff
        return res
