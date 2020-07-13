import os
from random import randint

import torch
import numpy as np
from torch import Tensor

from piepline.train import Trainer
from piepline import events_container
from piepline.train_config.metrics import AbstractMetric
from piepline.monitoring.hub import MonitorHub
from piepline.train import DecayingLR
from piepline.train_config.train_config import BaseTrainConfig
from piepline.train_config.stages import TrainStage, ValidationStage
from piepline.train_config.metrics_processor import MetricsProcessor
from piepline.utils.fsm import FileStructManager
from tests.common import UseFileStructure
from tests.data_processor_test import SimpleModel
from tests.data_producer_test import TestDataProducer

__all__ = ['TrainTest']


class SimpleLoss(torch.nn.Module):
    def forward(self, output, target):
        return output / target


class DummyMetric(AbstractMetric):
    def __init__(self):
        super().__init__('dummy_metric')

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        return float(torch.mean(output - target).numpy())


class TrainTest(UseFileStructure):
    def test_base_ops(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()

        trainer = Trainer(BaseTrainConfig(model, [], torch.nn.L1Loss(), torch.optim.SGD(model.parameters(), lr=1)),
                          fsm)
        with self.assertRaises(Trainer.TrainerException):
            trainer.train()

    def test_train(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))])),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]))]
        Trainer(BaseTrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm) \
            .set_epoch_num(1).train()

    def test_decaynig_lr(self):
        step_num = 0

        def target_value_clbk() -> float:
            return 1 / step_num

        lr = DecayingLR(0.1, 0.5, 3, target_value_clbk)
        old_val = None
        for i in range(1, 30):
            step_num = i
            value = lr.value()
            if old_val is None:
                old_val = value
                continue

            self.assertAlmostEqual(value, old_val, delta=1e-6)
            old_val = value

        step_num = 0

        def target_value_clbk() -> float:
            return 1

        lr = DecayingLR(0.1, 0.5, 3, target_value_clbk)
        old_val = None
        for i in range(1, 30):
            step_num = i
            value = lr.value()
            if old_val is None:
                old_val = value
                continue

            if i % 3 == 0:
                self.assertAlmostEqual(value, old_val * 0.5, delta=1e-6)
            old_val = value

        val = randint(1, 1000)
        lr.set_value(val)
        self.assertEqual(val, lr.value())

    def test_lr_decaying(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                                  metrics_processor)]
        trainer = Trainer(BaseTrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(10)

        def target_value_clbk() -> float:
            return 1

        trainer.enable_lr_decaying(0.5, 3, target_value_clbk)
        trainer.train()

        self.assertAlmostEqual(trainer.data_processor().get_lr(), 0.1 * (0.5 ** 3), delta=1e-6)

    def test_savig_states(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor)]
        trainer = Trainer(BaseTrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(3)

        checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'last', 'last_checkpoint.zip')

        def on_epoch_end():
            self.assertTrue(os.path.exists(checkpoint_file))
            os.remove(checkpoint_file)

        events_container.event(trainer, "EPOCH_END").add_callback(lambda x: on_epoch_end())
        trainer.train()

    def test_savig_best_states(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor)]
        trainer = Trainer(BaseTrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(3).enable_best_states_saving(lambda: np.mean(stages[0].get_losses()))

        checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'last', 'last_checkpoint.zip')
        best_checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'best', 'best_checkpoint.zip')

        class Val:
            def __init__(self):
                self.v = None

        first_val = Val()

        def on_epoch_end(val):
            if val.v is not None and np.mean(stages[0].get_losses()) < val.v:
                self.assertTrue(os.path.exists(best_checkpoint_file))
                os.remove(best_checkpoint_file)
                val.v = np.mean(stages[0].get_losses())
                return

            val.v = np.mean(stages[0].get_losses())

            self.assertTrue(os.path.exists(checkpoint_file))
            self.assertFalse(os.path.exists(best_checkpoint_file))
            os.remove(checkpoint_file)

        events_container.event(trainer, "EPOCH_END").add_callback(lambda x: on_epoch_end(first_val))
        trainer.train()

    def test_events(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        stage = TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]))
        trainer = Trainer(BaseTrainConfig(model, [stage], SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(3).enable_best_states_saving(lambda: np.mean(stage.get_losses()))

        metrics_processor = MetricsProcessor().subscribe_to_stage(stage)
        metrics_processor.add_metric(DummyMetric())

        with MonitorHub(trainer) as mh:
            def on_epoch_start(local_trainer: Trainer):
                self.assertIs(local_trainer, trainer)

            def on_epoch_end(local_trainer: Trainer):
                self.assertIs(local_trainer, trainer)
                self.assertEqual(20, local_trainer.train_config().stages()[0].get_losses().size)
                self.assertEqual(0, local_trainer.train_config().stages()[0].metrics_processor().get_metrics()['metrics'][0].get_values().size)

            def on_best_state_achieved(local_trainer: Trainer):
                self.assertIs(local_trainer, trainer)

            mh.subscribe2metrics_processor(metrics_processor)

            events_container.event(trainer, 'EPOCH_START').add_callback(on_epoch_start)
            events_container.event(trainer, 'EPOCH_END').add_callback(on_epoch_end)
            events_container.event(trainer, 'BEST_STATE_ACHIEVED').add_callback(on_best_state_achieved)

            trainer.train()

            self.assertEqual(None, trainer.train_config().stages()[0].get_losses())
