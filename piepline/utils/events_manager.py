class Event:
    def __init__(self, obj: object):
        self._callbacks = []
        self._object = obj

    def add_callback(self, clbk: callable):
        self._callbacks.append(clbk)

    def __call__(self):
        for clbk in self._callbacks:
            clbk(self._object)
