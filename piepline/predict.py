"""
The main module for run inference
"""
from abc import ABCMeta

from torch.nn import Module
from tqdm import tqdm
import torch

from piepline.utils.checkpoints_manager import CheckpointsManager
from piepline.data_producer.data_producer import DataProducer
from piepline.utils.fsm import FileStructManager
from piepline.data_processor.data_processor import DataProcessor

__all__ = ['Predictor', 'DataProducerPredictor']


class BasePredictor(metaclass=ABCMeta):
    def __init__(self, model: Module, checkpoints_manager: CheckpointsManager):
        self._data_processor = DataProcessor(model)

        checkpoints_manager.unpack()
        checkpoints_manager.load_model_weights(model)
        checkpoints_manager.pack()


class Predictor(BasePredictor):
    """
    Predictor run inference by training parameters

    :param model: model object, used for predict
    :param fsm: :class:`FileStructManager` object
    """

    def __init__(self, model: Module, checkpoints_manager: CheckpointsManager):
        super().__init__(model, checkpoints_manager)

    def predict(self, data: torch.Tensor or dict):
        """
        Predict ine data

        :param data: data as :class:`torch.Tensor` or dict with key ``data``
        :return: processed output
        :rtype: model output type
        """
        return self._data_processor.predict(data)


class DataProducerPredictor(BasePredictor):
    def __init__(self, model: Module, fsm: FileStructManager, checkpoints_manager: CheckpointsManager):
        super().__init__(model, fsm, checkpoints_manager)

    def predict(self, data_producer: DataProducer, callback: callable) -> None:
        """
        Run prediction iterates by ``data_producer``

        :param data_producer: :class:`DataProducer` object
        :param callback: callback, that call for every data prediction and get it's result as parameter
        """
        loader = data_producer.get_loader()

        for img in tqdm(loader):
            callback(self._data_processor.predict(img))
            del img
