__version__ = '0.2.2'

from utils.events_system import EventsContainer

events_container = EventsContainer()

from .data_producer import *
from .data_processor import *
from .train_config import *
from .utils import *
from .monitoring import MonitorHub, AbstractMonitor, ConsoleMonitor
from .train import Trainer
from .predict import Predictor
