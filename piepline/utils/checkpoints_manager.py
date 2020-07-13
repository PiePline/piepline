import os
from zipfile import ZipFile

from piepline.utils.fsm import FolderRegistrable, FileStructManager

__all__ = ['CheckpointsManager']


class CheckpointsManager(FolderRegistrable):
    """
    Class that manage checkpoints for DataProcessor.

    All states pack to zip file. It contains few files: model weights, optimizer state, data processor state

    :param fsm: :class:'FileStructureManager' instance
    :param prefix: prefix of saved and loaded files
    """

    class SMException(Exception):
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
            raise self.SMException("Checkpoints dir doesn't exists: [{}]".format(self._checkpoints_dir))

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
            raise self.SMException("Some files doesn't exists: [{}]".format(';'.join(files)))

    def _get_gir(self) -> str:
        return os.path.join('checkpoints', self._prefix)

    def _get_name(self) -> str:
        return 'CheckpointsManager' + self._prefix
