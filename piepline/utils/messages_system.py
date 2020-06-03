from typing import Any, List


class Message:
    def __init__(self):
        self._vals = []

    def write(self, val: Any) -> None:
        self._vals.append(val)

    def read(self) -> List[Any]:
        yield self._vals
        self._vals = []


class MessageReceiver:
    class MessageReceiverException(Exception):
        def __init__(self, msg):
            super.__init__(msg)

    def __init__(self):
        self._messages = {}

    def _add_message(self, name: str) -> None:
        if name in self._messages:
            if name in self._messages:
                raise MessageReceiver.MessageReceiverException("Event '{}' also exist".format(name))
            self._messages[name] = Message()

    def message(self, name: str) -> Message:
        return self._messages[name]