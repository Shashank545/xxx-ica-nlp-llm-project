import logging

from utils.metaclasses import Singleton


class NewLineFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None, style='{', *args, **kwargs):
        super().__init__(fmt, datefmt, style, *args, **kwargs)

    def format(self, record):
        record.msecs = record.msecs * 1000
        msg = logging.Formatter.format(self, record)
        if record.message != '':
            parts = msg.split(record.message)
            msg = msg.replace('\n', '\n' + parts[0])

        return msg


class FusionServicesLogger(logging.Logger, metaclass=Singleton):
    def __init__(self, name, level):
        stream_handler = logging.StreamHandler()
        formatter = NewLineFormatter(
            fmt='{asctime}.{msecs:.0f}|{levelname}|{funcName}:{lineno}| {msg}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{',
        )

        super().__init__(name=name, level=level)

        stream_handler.setFormatter(formatter)
        self.addHandler(stream_handler)
        self.propagate = False
