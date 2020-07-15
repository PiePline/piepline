import json
import os
from zipfile import ZipFile

import torch
from torch.nn import Module

from piepline import events_container
from piepline.utils.fsm import FolderRegistrable, FileStructManager
from piepline.train import Trainer
from piepline.train_config.stages import AbstractStage
from piepline.utils.events_system import Event

__all__ = ['CheckpointsManager', 'BestStateDetector']


class CheckpointsManager(FolderRegistrable):
    """
    Class that manage checkpoints for DataProcessor.

    All states pack to zip file. It contains few files: model weights, optimizer state, data processor state

    :param fsm: :class:'FileStructureManager' instance
    :param prefix: prefix of saved and loaded files
    """

    class CMException(Exception):
        """
        Exception for :class:`CheckpointsManager`
        """

        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, fsm: 'FileStructManager', prefix: str = None):
        super().__init__(fsm)

        self._prefix = prefix if prefix is not None else 'last'
        fsm.register_dir(self)
        self._checkpoints_dir = fsm.get_path(self, create_if_non_exists=True, check=False)

        if (prefix is None) and (not (os.path.exists(self._checkpoints_dir) and os.path.isdir(self._checkpoints_dir))):
            raise self.CMException("Checkpoints dir doesn't exists: [{}]".format(self._checkpoints_dir))

        self._weights_file = os.path.join(self._checkpoints_dir, 'weights.pth')
        self._state_file = os.path.join(self._checkpoints_dir, 'state.pth')
        self._checkpoint_file = self._compile_path(self._checkpoints_dir, 'checkpoint.zip')
        self._trainer_file = os.path.join(self._checkpoints_dir, 'trainer.json')

        if not fsm.in_continue_mode() and os.path.exists(self._weights_file) and os.path.exists(self._state_file) and \
                os.path.isfile(self._weights_file) and os.path.isfile(self._state_file):
            prev_prefix = self._prefix
            self._prefix = "prev_start"
            self.pack()
            self._prefix = prev_prefix

    def subscribe2trainer(self, trainer: Trainer) -> 'CheckpointsManager':
        events_container.event(trainer, 'EPOCH_END').add_callback(self.save_trainer_state)
        return self

    def unpack(self) -> None:
        """
        Unpack state files
        """
        with ZipFile(self._checkpoint_file, 'r') as zipfile:
            zipfile.extractall(self._checkpoints_dir)

        self._check_files([self._weights_file, self._state_file, self._trainer_file])

    def clear_files(self) -> None:
        """
        Clear unpacked files
        """

        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        rm_file(self._weights_file)
        rm_file(self._state_file)
        rm_file(self._trainer_file)

    def pack(self) -> None:
        """
        Pack all files in zip
        """

        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        def rename_file(file: str):
            target = file + ".old"
            rm_file(target)
            if os.path.exists(file) and os.path.isfile(file):
                os.rename(file, target)

        self._check_files([self._weights_file, self._state_file])

        rename_file(self._checkpoint_file)
        with ZipFile(self._checkpoint_file, 'w') as zipfile:
            zipfile.write(self._weights_file, os.path.basename(self._weights_file))
            zipfile.write(self._state_file, os.path.basename(self._state_file))
            zipfile.write(self._trainer_file, os.path.basename(self._trainer_file))

        self.clear_files()

    def optimizer_state_file(self) -> str:
        """
        Get optimizer state file path

        :return: path
        """
        return self._state_file

    def weights_file(self) -> str:
        """
        Get model weights file path

        :return: path
        """
        return self._weights_file

    def trainer_file(self) -> str:
        """
        Get trainer state file path

        :return: path
        """
        return self._trainer_file

    def save_trainer_state(self, trainer: Trainer) -> float or None:
        with open(self.trainer_file(), 'w') as out:
            json.dump({'last_epoch': trainer.cur_epoch_id()}, out)

        self.save_model_weights(trainer.data_processor().model())
        trainer.data_processor().save_state(self.optimizer_state_file())
        self.pack()

    def load_trainer_state(self, trainer: Trainer):
        self.unpack()

        with open(self.trainer_file(), 'r') as file:
            last_epoch = json.load(file)['last_epoch']
        trainer.set_cur_epoch(last_epoch + 1)

        self.load_data_processor(trainer.data_processor())
        self.load_model_weights(trainer.data_processor().model())

        self.pack()

    def _compile_path(self, directory: str, file: str) -> str:
        """
        Internal method for compile result file name

        :return: path to result file
        """
        return os.path.join(directory, (self._prefix + "_" if self._prefix is not None else "") + file)

    def _check_files(self, files) -> None:
        """
        Internal method for checking files for condition of existing

        :param files: list of files pathes to check
        :raises: SMException
        """
        failed = []
        for f in files:
            if not (os.path.exists(f) and os.path.isfile(f)):
                failed.append(f)

        if len(failed) > 0:
            raise self.CMException("Some files doesn't exists: [{}]".format(';'.join(files)))

    def _get_gir(self) -> str:
        return os.path.join('checkpoints', self._prefix)

    def _get_name(self) -> str:
        return 'CheckpointsManager' + self._prefix

    def load_model_weights(self, model: Module, weights_file: str = None) -> None:
        """
        Load weight from checkpoint
        """
        if weights_file is not None:
            file = weights_file
        else:
            if model is None:
                raise self.CMException('No weights file or CheckpointsManagement specified')
            file = self.weights_file()
        print("Model inited by file:", file, end='; ')
        pretrained_weights = torch.load(file)
        print("dict len before:", len(pretrained_weights), end='; ')
        processed = {}
        model_state_dict = model.state_dict()
        for k, v in pretrained_weights.items():
            if k.split('.')[0] == 'module' and not isinstance(model, torch.nn.DataParallel):
                k = '.'.join(k.split('.')[1:])
            elif isinstance(model, torch.nn.DataParallel) and k.split('.')[0] != 'module':
                k = 'module.' + k
            if k in model_state_dict:
                if v.device != model_state_dict[k].device:
                    v.to(model_state_dict[k].device)
                processed[k] = v

        model.load_state_dict(processed)
        print("dict len after:", len(processed))

    def save_model_weights(self, model: Module, weights_file: str = None) -> None:
        """
        Serialize weights to file
        """
        state_dict = model.state_dict()
        if weights_file is None:
            torch.save(state_dict, self.weights_file())
        else:
            torch.save(state_dict, weights_file)

    def load_data_processor(self, data_processor: 'TrainDataProcessor') -> None:
        """
        Load state of model, optimizer and TrainDataProcessor from checkpoint
        """
        print("Optimizer inited by file:", self.optimizer_state_file(), end='; ')
        state = torch.load(self.optimizer_state_file())
        print('state dict len before:', len(state), end='; ')
        state = {k: v for k, v in state.items() if k in data_processor.optimizer().state_dict()}
        print('state dict len after:', len(state), end='; ')
        data_processor.optimizer().load_state_dict(state)
        print('done')


class BestStateDetector:
    def __init__(self, trainer: Trainer):
        self._rules, self._prev_states = [], None
        self._best_state_achieved_event = events_container.add_event("BEST_STATE_ACHIEVED", Event(self))

        events_container.event(trainer, 'TRAIN_DONE').add_callback(lambda t: self.reset())

    def add_rule(self, rule: callable) -> 'BestStateDetector':
        self._rules.append(rule)
        return self

    def subscribe2stage(self, stage: AbstractStage) -> 'BestStateDetector':
        events_container.event(stage, 'EPOCH_END').add_callback(lambda t: self.check_best_state_achieved())
        return self

    def check_best_state_achieved(self) -> None:
        if self._prev_states is None:
            self._prev_states = []
            for rule in self._rules:
                self._prev_states.append(rule())
            return

        new_states = []
        for rule, prev_state in zip(self._rules, self._prev_states):
            cur_state = rule()
            if prev_state < cur_state:
                return
            new_states.append(cur_state)

        self._prev_states = new_states
        self._best_state_achieved_event()

    def reset(self):
        self._rules, self._prev_states = [], None
