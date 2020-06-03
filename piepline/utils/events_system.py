from typing import List


class Event:
    def __init__(self, obj: object):
        self._callbacks = []
        self._object = obj

    def add_callback(self, clbk: callable):
        self._callbacks.append(clbk)

    def __call__(self):
        for clbk in self._callbacks:
            clbk(self._object)


class EventExistsException(Exception):
    def __init__(self, msg):
        super.__init__(msg)


class EventsContainer:
    def __init__(self):
        self._events = {}

    def _add_event(self, name: str):
        if name in self._events:
            raise EventExistsException("Event '{}' also exist".format(name))
        self._events[name] = Event(self)

    def event(self, name: str) -> Event:
        return self._events[name]

    def events_names(self) -> List[str]:
        return list(self._events.keys())
