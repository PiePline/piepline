"""
The main module for training process
"""
import torch

from piepline import events_container
from piepline.train_config.train_config import BaseTrainConfig
from piepline.data_processor.data_processor import TrainDataProcessor
from piepline.utils.fsm import FileStructManager
from piepline.utils.events_system import Event
from piepline.utils.messages_system import MessageReceiver

__all__ = ['Trainer']


class LearningRate:
    """
    Basic learning rate class
    """

    def __init__(self, value: float):
        self._value = value

    def value(self) -> float:
        """
        Get value of current learning rate

        :return: current value
        """
        return self._value

    def set_value(self, value) -> None:
        """
        Set lr value

        :param value: lr value
        """
        self._value = value


class DecayingLR(LearningRate):
    """
    This class provide lr decaying by defined metric value (by :arg:`target_value_clbk`).
    If metric value doesn't update minimum after defined number of steps (:arg:`patience`) - lr was decaying
    by defined coefficient (:arg:`decay_coefficient`).

    :param start_value: start value
    :param decay_coefficient: coefficient of decaying
    :param patience: steps before decay
    :param target_value_clbk: callable, that return target value for lr decaying
    """

    def __init__(self, start_value: float, decay_coefficient: float, patience: int, target_value_clbk: callable):
        super().__init__(start_value)

        self._decay_coefficient = decay_coefficient
        self._patience = patience
        self._cur_step = 1
        self._target_value_clbk = target_value_clbk
        self._cur_min_target_val = None

    def value(self) -> float:
        """
        Get value of current learning rate

        :return: learning rate value
        """
        metric_val = self._target_value_clbk()
        if metric_val is None:
            return self._value

        if self._cur_min_target_val is None:
            self._cur_min_target_val = metric_val

        if metric_val < self._cur_min_target_val:
            self._cur_step = 1
            self._cur_min_target_val = metric_val

        if self._cur_step > 0 and (self._cur_step % self._patience) == 0:
            self._value *= self._decay_coefficient
            self._cur_min_target_val = None
            self._cur_step = 1
            return self._value

        self._cur_step += 1
        return self._value

    def set_value(self, value):
        self._value = value
        self._cur_step = 0
        self._cur_min_target_val = None


class Trainer(MessageReceiver):
    """
    Class, that run drive process.

    Trainer get list of training stages and every epoch loop over it.

    Training process looks like:

    .. highlight:: python
    .. code-block:: python

        for epoch in epochs_num:
            for stage in training_stages:
                stage.run()
                monitor_hub.update_metrics(stage.metrics_processor().get_metrics())
            save_state()
            on_epoch_end_callback()

    :param train_config: :class:`TrainConfig` object
    :param fsm: :class:`FileStructManager` object
    :param device: device for training process
    """

    class TrainerException(Exception):
        def __init__(self, msg):
            super().__init__()
            self._msg = msg

        def __str__(self):
            return self._msg

    def __init__(self, train_config: BaseTrainConfig, fsm: FileStructManager, device: torch.device = None):
        MessageReceiver.__init__(self)

        self._fsm = fsm

        self.__epoch_num, self._cur_epoch_id = 100, 0

        self._train_config = train_config
        self._data_processor = TrainDataProcessor(self._train_config, device)
        self._lr = LearningRate(self._data_processor.get_lr())

        self._epoch_end_event = events_container.add_event('EPOCH_END', Event(self))
        self._epoch_start_event = events_container.add_event('EPOCH_START', Event(self))
        self._train_done_event = events_container.add_event('TRAIN_DONE', Event(self))

        self._add_message('NEED_STOP')

    def set_epoch_num(self, epoch_number: int) -> 'Trainer':
        """
        Define number of epoch for training. One epoch - one iteration over all train stages

        :param epoch_number: number of training epoch
        :return: self object
        """
        self.__epoch_num = epoch_number
        return self

    def enable_lr_decaying(self, coeff: float, patience: int, target_val_clbk: callable) -> 'Trainer':
        """
        Enable rearing rate decaying. Learning rate decay when `target_val_clbk` returns doesn't update
        minimum for `patience` steps

        :param coeff: lr decay coefficient
        :param patience: number of steps
        :param target_val_clbk: callback which returns the value that is used for lr decaying
        :return: self object
        """
        self._lr = DecayingLR(self._data_processor.get_lr(), coeff, patience, target_val_clbk)
        return self

    def cur_epoch_id(self) -> int:
        """
        Get current epoch index
        """
        return self._cur_epoch_id

    def set_cur_epoch(self, idx: int) -> 'Trainer':
        self._cur_epoch_id = idx
        return self

    def train(self) -> None:
        """
        Run training process
        """
        if len(self._train_config.stages()) < 1:
            raise self.TrainerException("There's no sages for training")

        start_epoch_idx = self._cur_epoch_id

        self._connect_stages_to_events()

        for epoch_idx in range(start_epoch_idx, self.__epoch_num + start_epoch_idx):
            if True in self.message('NEED_STOP').read():
                break

            self._cur_epoch_id = epoch_idx
            self._epoch_start_event()

            for stage in self._train_config.stages():
                stage.run(self._data_processor)

            self._data_processor.update_lr(self._lr.value())
            self._epoch_end_event()

        self._train_done_event()

    def _update_losses(self) -> None:
        """
        Update loses procedure
        """
        losses = {}
        for stage in self._train_config.stages():
            if stage.get_losses() is not None:
                losses[stage.name()] = stage.get_losses()
        self.monitor_hub.update_losses(losses)

    def data_processor(self) -> TrainDataProcessor:
        """
        Get data processor object

        :return: data processor
        """
        return self._data_processor

    def train_config(self) -> BaseTrainConfig:
        """
        Get train config

        :return: TrainConfig object
        """
        return self._train_config

    def _connect_stages_to_events(self):
        for stage in self._train_config.stages():
            self._epoch_end_event.add_callback(lambda x: stage.on_epoch_end())
