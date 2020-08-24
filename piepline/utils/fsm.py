"""
This module contains all classes, that work with file structure

* :class:`FileStructManager` provide all modules registration
* :class:`CheckpointsManager` provide checkpoints management
"""

import os
from abc import ABCMeta, abstractmethod

__all__ = ['FileStructManager', 'FolderRegistrable', 'MultipleFSM']


class FolderRegistrable(metaclass=ABCMeta):
    """
    Abstract class for implement classes, that use folders

    :param fsm: FileStructureManager class instance
    """

    @abstractmethod
    def __init__(self, fsm: 'FileStructManager'):
        pass

    @abstractmethod
    def _get_gir(self) -> str:
        """
        Get directory path to register

        :return: path
        """

    @abstractmethod
    def _get_name(self) -> str:
        """
        Get name of registrable object

        :return: name
        """


class FileStructManager:
    """
    Class, that provide directories registration in base directory.

    All modules, that use file structure under base directory should register their paths in this class by pass module
    to method :meth:`register_dir`.
    If directory also registered registration method will raise exception :class:`FSMException`

    :param base_dir: path to directory with checkpoints
    :param is_continue: is `FileStructManager` used for continue training or predict
    :param exists_ok: if `True` - all checks for existing directories will be disabled
    """

    class FSMException(Exception):
        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    class _Folder:
        """
        Internal class, that implements logic for single registrable directory

        :param path: path to directory
        :param fsm: :class:`FileStructManager` object
        """

        def __init__(self, path: str, fsm: 'FileStructManager'):
            self._path = path
            self._fsm = fsm

            self._path_first_request = True

        def get_path_for_check(self) -> str:
            """
            Get folder path without any checking for existing

            :return: path
            """
            return self._path

        def _create_directories(self) -> None:
            """
            Internal method that create directory if this not exists and FileStructManager not in continue mode
            """
            if self._fsm._is_continue:
                return
            if not (os.path.exists(self._path) and os.path.isdir(self._path)):
                os.makedirs(self._path, exist_ok=True)

        def get_path(self, create_if_non_exists: bool = True) -> str:
            """
            Get folder path. This method create directory if it's not exists (if param ``create_if_non_exists == True``)

            :param create_if_non_exists: is need to create directory if it's doesn't exists
            :return: directory path
            """
            if create_if_non_exists and self._path_first_request:
                self._create_directories()
                self._path_first_request = False
            return self._path

        def check_path(self) -> None:
            """
            Check that directory doesn't contains any files

            :raises: :class:`FileStructManager.FSMException`
            """
            if os.path.exists(self._path) and os.path.isdir(self._path):
                if os.listdir(self._path):
                    raise self._fsm.FSMException("Checkpoint directory already exists [{}]".format(self._path))

    def __init__(self, base_dir: str, is_continue: bool, exists_ok: bool = False):
        self._dirs = {}
        self._is_continue = is_continue
        self._base_dir = base_dir
        self._exist_ok = exists_ok

    def register_dir(self, obj: FolderRegistrable, check_name_registered: bool = False, check_dir_registered: bool = True) -> None:
        """
        Register directory in file structure

        :param obj: object to registration
        :param check_name_registered: is need to check if object name also registered
        :param check_dir_registered: is need to check if object path also registered
        :raise FileStructManager: if path or object name also registered and if path also exists (in depends of optional parameters values)
        """
        path = self._compile_path(obj)

        if check_dir_registered:
            for n, f in self._dirs.items():
                if f.get_path_for_check() == path:
                    raise self.FSMException("Path {} already registered!".format(path))

        if check_name_registered:
            if obj._get_name() in self._dirs:
                raise self.FSMException("Object {} already registered!".format(obj._get_name()))

        self._dirs[obj._get_name()] = self._Folder(path, self)
        if not self._exist_ok and not self._is_continue:
            self._dirs[obj._get_name()].check_path()

    def get_path(self, obj: FolderRegistrable, create_if_non_exists: bool = False, check: bool = True) -> str:
        """
        Get path of registered object

        :param obj: object
        :param create_if_non_exists: is need to create object's directory if it doesn't exists
        :param check: is need to check object's directory existing
        :return: path to directory
        :raise FSMException: if directory exists and ``check == True``
        """
        directory = self._dirs[obj._get_name()]
        if not self._exist_ok and not self._is_continue and check:
            directory.check_path()
        return directory.get_path(create_if_non_exists)

    def in_continue_mode(self) -> bool:
        """
        Is FileStructManager in continue mode

        :return: True if in continue
        """
        return self._is_continue

    def _compile_path(self, obj: FolderRegistrable) -> str:
        return os.path.join(self._base_dir, obj._get_gir())


class MultipleFSM(FileStructManager):
    def __init__(self, base_dir: str, is_continue: bool, exists_ok: bool = False):
        super().__init__(base_dir, is_continue, exists_ok)
        self._cur_experiment_name = None
        self._objects_nums = {}

    def set_namespace(self, name: str) -> 'MultipleFSM':
        self._cur_experiment_name = name
        return self

    def _compile_path(self, obj: FolderRegistrable) -> str:
        if obj._get_name() not in self._objects_nums:
            self._objects_nums[obj._get_name()] = 0
        else:
            self._objects_nums[obj._get_name()] += 1
        return os.path.join(self._base_dir, str(self._objects_nums[obj._get_name()]), obj._get_gir())
