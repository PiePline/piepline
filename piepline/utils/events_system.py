from typing import List


class Event:
    def __init__(self, obj: object):
        self._callbacks = []
        self._object = obj

    def object(self):
        return self._object

    def add_callback(self, clbk: callable):
        self._callbacks.append(clbk)

    def __call__(self):
        for clbk in self._callbacks:
            clbk(self._object)


class EventExistsException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class EventsContainer:
    def __init__(self):
        self._events = {}

    def add_event(self, name: str, event: Event) -> Event:
        if name in self._events:
            raise EventExistsException("Event '{}' also exist".format(name))
        if event.object() not in self._events:
            self._events[event.object()] = {}
        self._events[event.object()][name] = event
        return event

    def event(self, transmitter: object, name: str) -> Event:
        return self._events[transmitter][name]

    def events_names(self) -> List[str]:
        return list(self._events.keys())
