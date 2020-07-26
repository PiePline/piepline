from piepline.monitoring.monitors import AbstractMetricsMonitor
from piepline import events_container
from piepline.train import Trainer
from piepline.train_config.metrics_processor import MetricsProcessor

__all__ = ['MonitorHub']


class MonitorHub:
    """
    Aggregator of monitors. This class collect monitors and provide unified interface to it's
    """

    def __init__(self, trainer: Trainer):
        self.monitors = []
        events_container.event(trainer, 'EPOCH_START').add_callback(lambda t: self.set_epoch_num(t.cur_epoch_id()))

    def subscribe2metrics_processor(self, metrics_processor: MetricsProcessor) -> 'MonitorHub':
        events_container.event(metrics_processor, "BEFORE_METRICS_RESET").add_callback(lambda mp: self.update_metrics(mp.get_metrics()))
        return self

    def set_epoch_num(self, epoch_num: int) -> None:
        """
        Set current epoch num

        :param epoch_num: num of current epoch
        """
        for m in self.monitors:
            m.set_epoch_num(epoch_num)

    def add_monitor(self, monitor: AbstractMetricsMonitor) -> 'MonitorHub':
        """
        Connect monitor to hub

        :param monitor: :class:`AbstractMonitor` object
        :return:
        """
        self.monitors.append(monitor)
        return self

    def update_metrics(self, metrics: {}) -> None:
        """
        Update metrics in all monitors

        :param metrics: metrics dict with keys 'metrics' and 'groups'
        """
        for m in self.monitors:
            m.update_metrics(metrics)

    def update_losses(self, losses: {}) -> None:
        """
        Update monitor

        :param losses: losses values with keys 'train' and 'validation'
        """
        for m in self.monitors:
            m.update_losses(losses)

    def register_event(self, text: str) -> None:
        for m in self.monitors:
            m.register_event(text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.monitors:
            m.__exit__(exc_type, exc_val, exc_tb)
