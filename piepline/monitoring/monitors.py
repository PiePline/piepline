import json
import os
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from piepline.utils.fsm import FolderRegistrable, FileStructManager
from piepline.train_config.metrics import MetricsGroup, AbstractMetric
from piepline.utils.utils import dict_recursive_bypass

__all__ = ['AbstractMetricsMonitor', 'ConsoleLossMonitor', 'FileLogMonitor']


class AbstractMonitor(metaclass=ABCMeta):
    def __init__(self):
        self._epoch_num = 0

    def set_epoch_num(self, epoch_num: int) -> None:
        """
        Set current epoch num

        :param epoch_num: num of current epoch
        """
        self._epoch_num = epoch_num

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AbstractLossMonitor(AbstractMonitor, metaclass=ABCMeta):
    """
    Basic class for every loss monitor
    """

    def update_losses(self, losses: {}) -> None:
        """
        Update losses on monitor

        :param losses: losses values dict with keys is names of stages in train pipeline (e.g. [train, validation])
        """
        pass

    @staticmethod
    def _iterate_by_losses(losses: {}, callback: callable) -> None:
        """
        Internal method for unify iteration by losses dict

        :param losses: dic of losses
        :param callback: callable, that call for every loss value and get params loss_name and loss_values: ``callback(name: str, values: np.ndarray)``
        """
        for m, v in losses.items():
            callback(m, v)


class AbstractMetricsMonitor(AbstractMonitor, metaclass=ABCMeta):
    """
    Basic class for every metrics monitor
    """

    def update_metrics(self, metrics: {}) -> None:
        """
        Update metrics on   monitor

        :param metrics: metrics dict with keys 'metrics' and 'groups'
        """
        for metric in metrics['metrics']:
            self._process_metric([], metric)

        for metrics_group in metrics['groups']:
            for metric in metrics_group.metrics():
                self._process_metric([metrics_group], metric)
            for group in metrics_group.groups():
                self._process_metric([metrics_group, group], metrics_group.name())

    @abstractmethod
    def _process_metric(self, path: List[MetricsGroup], metric: 'AbstractMetric'):
        """
        Internal method for process one metric

        Args:
            path (List[MetricsGroup]): list of parent metrics groups from root to current metric (except this metric)
        """


class ConsoleLossMonitor(AbstractLossMonitor):
    """
    Monitor, that used for write metrics to console.

    Output looks like: ``Epoch: [#]; train: [-1, 0, 1]; validation: [-1, 0, 1]``. This 3 numbers is [min, mean, max] values of
    training stage loss values
    """

    class ResStr:
        def __init__(self, start: str):
            self.res = start

        def append(self, string: str):
            self.res += string

        def __str__(self):
            return self.res[:len(self.res) - 1]

    def update_losses(self, losses: {}) -> None:
        def on_loss(name: str, values: np.ndarray, string) -> None:
            string.append(" {}: [{:4f}, {:4f}, {:4f}];".format(name, np.min(values), np.mean(values), np.max(values)))

        res_string = self.ResStr("Epoch: [{}];".format(self._epoch_num))
        self._iterate_by_losses(losses, lambda m, v: on_loss(m, v, res_string))
        print(res_string)


class FileLogMonitor(AbstractMetricsMonitor, AbstractLossMonitor, FolderRegistrable):
    """
    Monitor, used for logging metrics in files. It's write full log and can also write last metrics in separate file if required

    All output files in JSON format and stores in ``<base_dir_path>/monitors/metrics_log``

    :param fsm: :class:`FileStructManager` object
    """

    def __init__(self, fsm: FileStructManager):
        super().__init__()

        self._fsm = fsm
        self._fsm.register_dir(self)
        self._files = {}
        self._meta_file = None

    def _process_metric(self, path: List[MetricsGroup], metric: 'AbstractMetric'):
        cur_dir = self._fsm.get_path(self, create_if_non_exists=True, check=True)
        for metrics_grp in path:
            cur_dir = os.path.join(cur_dir, metrics_grp.name())

        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        cur_file_path = os.path.join(cur_dir, metric.name() + '.csv')
        with open(cur_file_path, 'w') as out:
            out.write("{}, {}\n".format(self._epoch_num, metric.get_value()))

        if cur_file_path not in self._files:
            self._files[cur_file_path] = metric.name()

            if self._meta_file is None:
                self._meta_file = os.path.join(cur_dir, 'meta.json')
                with open(self._meta_file, 'w') as meta_out:
                    json.dump(list(self._files.values()), meta_out)

    def load(self) -> dict:
        cur_dir = self._fsm.get_path(self, create_if_non_exists=False, check=True)
        with open(os.path.join(cur_dir, 'meta.json'), 'r') as meta_file:
            meta = json.load(meta_file)

        res = {}
        for path in meta:
            cur_path = os.path.join(cur_dir, path)
            for f in os.listdir(cur_path):
                res[path] = np.loadtxt(os.path.join(cur_path, f), delimiter=',')

        return res

    def _get_gir(self) -> str:
        return os.path.join('monitors', 'metrics_log')

    def _get_name(self) -> str:
        return 'FileLogMonitor'
