"""
This module contains Tensorboard monitor interface
"""

import os
from typing import List

import numpy as np
from torch.nn import Module

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("Can't import tensorboard. Try to install tensorboardX or update PyTorch version")

from piepline.monitoring.monitors import AbstractMetricsMonitor, AbstractLossMonitor
from piepline.train_config.metrics import AbstractMetric, MetricsGroup
from piepline.utils.fsm import FileStructManager, FolderRegistrable

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class TensorboardMonitor(AbstractMetricsMonitor, AbstractLossMonitor, FolderRegistrable):
    """
    Class, that manage metrics end events monitoring. It worked with tensorboard. Monitor get metrics after epoch ends and visualise it. Metrics may be float or np.array values. If
    metric is np.array - it will be shown as histogram and scalars (scalar plots contains mean valuse from array).

    :param fsm: file structure manager
    :param is_continue: is data processor continue training
    :param network_name: network name
    """

    def __init__(self, fsm: FileStructManager, is_continue: bool, network_name: str = None):
        super().__init__()
        self._writer = None
        self._txt_log_file = None

        fsm.register_dir(self)
        directory = fsm.get_path(self)
        if directory is None:
            return

        directory = os.path.join(directory, network_name) if network_name is not None else directory

        if not (fsm.in_continue_mode() or is_continue) and os.path.exists(directory) and os.path.isdir(directory):
            idx = 0
            tmp_dir = directory + "_v{}".format(idx)
            while os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
                idx += 1
                tmp_dir = directory + "_v{}".format(idx)
            directory = tmp_dir

        os.makedirs(directory, exist_ok=True)
        self._writer = SummaryWriter(directory)
        self._txt_log_file = open(os.path.join(directory, "log.txt"), 'a' if is_continue else 'w')

    def update_losses(self, losses: {}) -> None:
        """
        Update monitor

        :param losses: losses values with keys 'train' and 'validation'
        """
        if self._writer is None:
            return

        def on_loss(name: str, values: np.ndarray or dict) -> None:
            if isinstance(values, dict):
                self._writer.add_scalars('loss_{}'.format(name), {k: np.mean(v) for k, v in values.items()},
                                         global_step=self._epoch_num)
                for k, v in values.items():
                    self._writer.add_histogram('{}/loss_{}_hist'.format(name, k), np.clip(v, -1, 1).astype(np.float32),
                                               global_step=self._epoch_num, bins=np.linspace(-1, 1, num=11).astype(np.float32))
            else:
                self._writer.add_scalars('loss', {name: np.mean(values)}, global_step=self._epoch_num)
                self._writer.add_histogram('{}/loss_hist'.format(name), np.clip(values, -1, 1).astype(np.float32),
                                           global_step=self._epoch_num, bins=np.linspace(-1, 1, num=11).astype(np.float32))

        self._iterate_by_losses(losses, on_loss)

    def _process_metric(self, path: List[MetricsGroup], metric: 'AbstractMetric'):
        """
        Update console

        :param metrics: metrics
        """

        if self._writer is None:
            return

        def add_histogram(name: str, vals, step_num, bins):
            try:
                self._writer.add_histogram(name, vals, step_num, bins)
            except Exception:
                pass

        tag = '/'.join([p.name() for p in path] + [metric.name()])

        values = metric.get_values().astype(np.float32)
        if values.size > 0:
            self._writer.add_scalar(tag, float(np.mean(values)), global_step=self._epoch_num)
            add_histogram(tag + '_hist',
                          np.clip(values, metric.min_val(), metric.max_val()).astype(np.float32),
                          self._epoch_num, np.linspace(metric.min_val(), metric.max_val(), num=11).astype(np.float32))

    def update_scalar(self, name: str, value: float, epoch_idx: int = None) -> None:
        """
        Update scalar on tensorboard

        :param name: the classic tag for TensorboardX
        :param value: scalar value
        :param epoch_idx: epoch idx. If doesn't set - use last epoch idx stored in this class
        """
        self._writer.add_scalar(name, value, global_step=(epoch_idx if epoch_idx is not None else self._epoch_num))

    def visualize_model(self, model: Module, tensor) -> None:
        """
        Visualize model graph

        :param model: :class:`torch.nn.Module` object
        :param tensor: dummy input for trace model
        """
        self._writer.add_graph(model, tensor)

    def close(self):
        if self._txt_log_file is not None:
            self._txt_log_file.close()
            self._txt_log_file = None
            del self._txt_log_file
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            del self._writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_gir(self) -> str:
        return os.path.join('monitors', 'tensorboard')

    def _get_name(self) -> str:
        return 'Tensorboard'
