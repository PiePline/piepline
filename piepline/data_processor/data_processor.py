from typing import Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.nn import Module

from piepline.utils.utils import dict_recursive_bypass
from piepline.train_config.train_config import BaseTrainConfig


__all__ = ['DataProcessor', 'TrainDataProcessor']


class DataProcessor:
    """
    DataProcessor manage: model, data processing, device choosing

    Args:
        model (Module): model, that will be used for process data
        device (torch.device): what device pass data for processing
    """

    def __init__(self, model: Module, device: torch.device = None):
        self._checkpoints_manager = None
        self._model = model
        self._device = device

        self._pick_model_input = lambda data: data['data']
        self._data_preprocess = lambda data: data
        self._data_to_device = self._pass_object_to_device

    def model(self) -> Module:
        """
        Get current module
        """
        return self._model

    def predict(self, data: torch.Tensor or dict) -> object:
        """
        Make predict by data

        :param data: data as :class:`torch.Tensor` or dict with key ``data``
        :return: processed output
        :rtype: the model output type
        """
        self.model().eval()
        with torch.no_grad():
            output = self._model(self._data_to_device(self._data_preprocess(self._pick_model_input(data))))
        return output

    def set_data_to_device(self, data_to_device: callable) -> 'DataProcessor':
        self._data_to_device = data_to_device
        return self

    def set_pick_model_input(self, pick_model_input: callable) -> 'DataProcessor':
        """
        Set callback, that will get output from :mod:`DataLoader` and return model input.

        Default mode:

        .. highlight:: python
        .. code-block:: python

        lambda data: data['data']

        Args:
            pick_model_input (callable): pick model input callable. This callback need to get one parameter: dataset output

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            data_processor.set_pick_model_input(lambda data: data['data'])
            data_processor.set_pick_model_input(lambda data: data[0])
        """
        self._pick_model_input = pick_model_input
        return self

    def set_data_preprocess(self, data_preprocess: callable) -> 'DataProcessor':
        """
        Set callback, that will get output from :mod:`DataLoader` and return preprocessed data.
        For example may be used for pass data to device.

        Default mode:

        .. highlight:: python
        .. code-block:: python

        :meth:`_pass_data_to_device`

        Args:
            data_preprocess (callable): preprocess callable. This callback need to get one parameter: dataset output

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            from piepline.utils import dict_recursive_bypass
            data_processor.set_data_preprocess(lambda data: dict_recursive_bypass(data, lambda v: v.cuda()))
        """
        self._data_preprocess = data_preprocess
        return self

    def _pass_object_to_device(self, data) -> torch.Tensor or dict:
        """
        Internal method, that pass data to specified device
        :param data: data as any object type. If will passed to device if it's instance of :class:`torch.Tensor` or dict with key
        ``data``. Otherwise data will be doesn't changed
        :return: processed on target device
        """
        if self._device is None:
            return data

        if isinstance(data, dict):
            return dict_recursive_bypass(data, lambda v: v.to(self._device))
        elif isinstance(data, torch.Tensor):
            return data.to(self._device)
        else:
            return data


class TrainDataProcessor(DataProcessor):
    """
    TrainDataProcessor is make all of DataProcessor but produce training process.

    :param train_config: train config
    """

    class TDPException(Exception):
        def __init__(self, msg):
            self._msg = msg

        def __str__(self):
            return self._msg

    def __init__(self, train_config: 'BaseTrainConfig', device: torch.device = None):
        super().__init__(train_config.model(), device)

        self._pick_target = lambda data: data['target']
        self._target_preprocess = lambda data: data
        self._target_to_device = self._pass_object_to_device

        self._criterion = train_config.loss()
        self._optimizer = train_config.optimizer()

    def optimizer(self) -> Optimizer:
        return self._optimizer

    def predict(self, data, is_train=False) -> torch.Tensor or dict:
        """
        Make predict by data. If ``is_train`` is ``True`` - this operation will compute gradients. If
        ``is_train`` is ``False`` - this will work with ``model.eval()`` and ``torch.no_grad``

        :param data: data in dict
        :param is_train: is data processor need train on data or just predict
        :return: processed output
        :rtype: model return type
        """

        if is_train:
            self.model().train()
            output = self._model(self._data_to_device(self._data_preprocess(self._pick_model_input(data))))
        else:
            output = super().predict(data)

        return output

    def process_batch(self, batch: {}, is_train: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one batch of data

        Args:
            batch (dict): contains 'data' and 'target' keys. The values for key must be instance of torch.Tensor or dict
            is_train (bool): is batch process for train

        Returns:
            tuple of `class`:torch.Tensor of losses, predicts and targets with shape (N, ...) where N is batch size
        """
        if is_train:
            self._optimizer.zero_grad()

        res = self.predict(batch, is_train)
        target = self._target_to_device(self._target_preprocess(self._pick_target(batch)))
        loss = self._criterion(res, target)

        if is_train:
            loss.backward()
            self._optimizer.step()

        return loss, res, target

    def update_lr(self, lr: float) -> None:
        """
        Update learning rate straight to optimizer

        :param lr: target learning rate
        """
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """
        Get learning rate from optimizer
        """
        for param_group in self._optimizer.param_groups:
            return param_group['lr']

    def get_state(self) -> {}:
        """
        Get model and optimizer state dicts

        :return: dict with keys [weights, optimizer]
        """
        return {'weights': self._model.model().state_dict(), 'optimizer': self._optimizer.state_dict()}

    def save_state(self, path: str) -> None:
        """
        Save state of optimizer and perform epochs number
        """
        torch.save(self.optimizer().state_dict(), path)

    def set_pick_target(self, pick_target: callable) -> 'DataProcessor':
        """
        Set callback, that will get output from :mod:`DataLoader` and return target.

        Default mode:

        .. highlight:: python
        .. code-block:: python

        lambda data: data['target']

        Args:
            pick_target (callable): pick target callable. This callback need to get one parameter: dataset output

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            data_processor.set_pick_target(lambda data: data['target'])
            data_processor.set_pick_target(lambda data: data[1])
        """
        self._pick_target = pick_target
        return self

    def set_target_preprocess(self, target_preprocess: callable) -> 'DataProcessor':
        """
        Set callback, that will get output from :mod:`DataLoader` and return preprocessed target.
        For example may be used for pass target to device.

        Default mode:

        .. highlight:: python
        .. code-block:: python

        :meth:`_pass_target_to_device`

        Args:
            target_preprocess (callable): preprocess callable. This callback need to get one parameter: targetset output

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            from piepline.utils import dict_recursive_bypass
            target_processor.set_target_preprocess(lambda target: dict_recursive_bypass(target, lambda v: v.cuda()))
        """
        self._target_preprocess = target_preprocess
        return self

    def set_target_to_device(self, target_to_device: callable) -> 'DataProcessor':
        self._target_to_device = target_to_device
        return self
