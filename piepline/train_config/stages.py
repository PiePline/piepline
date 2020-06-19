from abc import ABCMeta, abstractmethod

from torch.utils.data.dataloader import DataLoader
import numpy as np

# from piepline.monitoring import MonitorHub
from piepline.data_producer.data_producer import DataProducer
from piepline.data_processor import DataProcessor, TrainDataProcessor
from piepline.train_config.metrics import MetricsProcessor
from piepline import events_container
from piepline.utils.events_system import Event

__all__ = ['AbstractStage', 'TrainStage', 'ValidationStage']


class AbstractStage(metaclass=ABCMeta):
    """
    Stage of training process. For example there may be 2 stages: train and validation.
    Every epochs in train loop is iteration by stages.

    :param name: name of stage
    """

    def __init__(self, name: str):
        self._name = name
        self._stage_end_event = Event(self)

    def name(self) -> str:
        """
        Get name of stage

        :return: name
        """
        return self._name

    @abstractmethod
    def _run(self, data_processor: DataProcessor) -> None:
        """
        Internal method with stage run implementation. This method was called in :meth:`run`
        """

    def run(self, data_processor: DataProcessor) -> None:
        """
        Run stage

        Args:
            data_processor (class:`DataProcessor`): data processor object
        """
        self._run(data_processor)
        self._stage_end_event()

    def get_losses(self) -> np.ndarray or None:
        """
        Get losses from this stage

        :return: array of losses or None if this stage doesn't need losses
        """
        return None

    def on_epoch_end(self) -> None:
        """
        Callback for train epoch end
        """
        pass


class StandardStage(AbstractStage):
    """
    Standard stage for train process.

    When call :meth:`run` it's iterate :meth:`process_batch` of data processor by data loader

    After stop iteration ValidationStage accumulate losses from :class:`DataProcessor`.

    :param data_producer: :class:`DataProducer` object
    """

    def __init__(self, stage_name: str, is_train: bool, data_producer: DataProducer):
        super().__init__(name=stage_name)
        self.data_loader = None
        self.data_producer = data_producer
        self._losses = None
        self._is_train = is_train

        self._epoch_end_event = events_container.add_event('EPOCH_END', Event(self))
        self._epoch_start_event = events_container.add_event('EPOCH_START', Event(self))

    def _run(self, data_processor: TrainDataProcessor) -> None:
        """
        Run stage. This iterate by DataProducer and show progress in stdout

        :param data_processor: :class:`DataProcessor` object
        """
        if self.data_loader is None:
            self.data_loader = self.data_producer.get_loader()

        self._run_internal(self.data_loader, self.name(), data_processor)

    def _run_internal(self, data_loader: DataLoader, name: str, data_processor: TrainDataProcessor):
        self._epoch_start_event()

        with tqdm(data_loader, desc=name, leave=False) as t:
            self._losses = None
            for batch in t:
                self._process_batch(batch, data_processor)
                t.set_postfix({'loss': '[{:4f}]'.format(np.mean(self._losses))})

        self._epoch_end_event()
        self._losses = None

    def _process_batch(self, batch, data_processor: TrainDataProcessor):
        cur_loss = data_processor.process_batch(batch, is_train=self._is_train)
        if self._losses is None:
            self._losses = cur_loss
        else:
            self._losses = np.append(self._losses, cur_loss)

    def get_losses(self) -> np.ndarray:
        """
        Get losses from this stage

        :return: array of losses
        """
        return self._losses

    def connect2monitor_hub(self, monitor_hub, metrics_processor: MetricsProcessor) -> 'StandardStage':
        events_container.event(self, 'EPOCH_END').add_callback(
            lambda stage: monitor_hub.update_metrics(metrics_processor.get_metrics()))
        events_container.event(self, 'EPOCH_END').add_callback(lambda stage: metrics_processor.reset_metrics())
        return self


class TrainStage(StandardStage):
    """
    Standard training stage

    When call :meth:`run` it's iterate :meth:`process_batch` of data processor by data loader with ``is_train=True`` flag.

    After stop iteration ValidationStage accumulate losses from :class:`DataProcessor`.

    :param data_producer: :class:`DataProducer` object
    :param name: name of stage. By default 'train'
    """

    class _HardNegativesTrainStage(StandardStage):
        def __init__(self, stage_name: str, data_producer: DataProducer, part: float):
            super().__init__(stage_name, True, data_producer)
            self._part = part

        def exec(self, data_processor: TrainDataProcessor, losses: np.ndarray, indices: []) -> None:
            num_losses = int(losses.size * self._part)
            idxs = np.argpartition(losses, -num_losses)[-num_losses:]
            self._run_internal(self.data_producer.get_loader([indices[i] for i in idxs]), self.name(), data_processor)

    def __init__(self, data_producer: DataProducer, name: str = 'train'):
        super().__init__(name, True, data_producer)
        self.hnm = None
        self.hn_indices = []
        self._dp_pass_indices_earlier = False

    def enable_hard_negative_mining(self, part: float) -> 'TrainStage':
        """
        Enable hard negative mining. Hard negatives was taken by losses values

        :param part: part of data that repeat after train stage
        :return: self object
        """

        if not 0 < part < 1:
            raise ValueError('Value of part for hard negative mining is out of range (0, 1)')
        self.hnm = self._HardNegativesTrainStage(self.name() + '_hnm', self.data_producer, part)
        self._dp_pass_indices_earlier = self.data_producer._is_passed_indices()
        self.data_producer.pass_indices(True)
        return self

    def disable_hard_negative_mining(self) -> 'TrainStage':
        """
        Enable hard negative mining.

        :return: self object
        """
        self.hnm = None
        if not self._dp_pass_indices_earlier:
            self.data_producer.pass_indices(False)
        return self

    def run(self, data_processor: TrainDataProcessor) -> None:
        """
        Run stage

        :param data_processor: :class:`TrainDataProcessor` object
        """
        super().run(data_processor)
        if self.hnm is not None:
            self.hnm.exec(data_processor, self._losses, self.hn_indices)
            self.hn_indices = []

    def _process_batch(self, batch, data_processor: TrainDataProcessor) -> None:
        """
        Internal method for process one bathc

        :param batch: batch
        :param data_processor: :class:`TrainDataProcessor` instance
        """
        if self.hnm is not None:
            self.hn_indices.append(batch['data_idx'])
        super()._process_batch(batch, data_processor)

    def on_epoch_end(self):
        """
        Method, that calls after every epoch
        """
        super().on_epoch_end()
        if self.hnm is not None:
            self.hnm.on_epoch_end()


class ValidationStage(StandardStage):
    """
    Standard validation stage.

    When call :meth:`run` it's iterate :meth:`process_batch` of data processor by data loader with ``is_tran=False`` flag.

    After stop iteration ValidationStage accumulate losses from :class:`DataProcessor`.

    :param data_producer: :class:`DataProducer` object
    :param name: name of stage. By default 'validation'
    """

    def __init__(self, data_producer: DataProducer, name: str = 'validation'):
        super().__init__(name, False, data_producer)
