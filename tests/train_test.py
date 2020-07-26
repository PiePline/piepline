import os
from random import randint

import torch
import numpy as np
from torch import Tensor

from piepline.train import Trainer
from piepline import events_container
from piepline.train_config.metrics import AbstractMetric, MetricsGroup
from piepline.monitoring.hub import MonitorHub
from piepline.train import DecayingLR
from piepline.train_config.train_config import BaseTrainConfig
from piepline.train_config.stages import TrainStage, ValidationStage
from piepline.train_config.metrics_processor import MetricsProcessor
from piepline.utils.fsm import FileStructManager
from piepline.utils.checkpoints_manager import CheckpointsManager, BestStateDetector

from tests.common import UseFileStructure, SimpleMetric
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
        return float(torch.mean(output - target).detach().cpu().numpy())


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
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))])),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]))]
        trainer = Trainer(BaseTrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(10)

        def target_value_clbk() -> float:
            return 1

        trainer.enable_lr_decaying(0.5, 3, target_value_clbk)
        trainer.train()

        self.assertAlmostEqual(trainer.data_processor().get_lr(), 0.1 * (0.5 ** 3), delta=1e-6)

    def test_saving_states(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stage = TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]))

        class Losses:
            def __init__(self):
                self.v = []
                self._fake_losses = [[i for _ in list(range(20))] for i in [5, 4, 0, 2, 1]]

            def on_stage_end(self, s: TrainStage):
                s._losses = self._fake_losses[0]
                del self._fake_losses[0]
                self.v.append(np.mean(s.get_losses()))

        losses = Losses()
        events_container.event(stage, 'EPOCH_END').add_callback(losses.on_stage_end)

        trainer = Trainer(BaseTrainConfig(model, [stage], SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(5)
        metrics_processor.subscribe_to_stage(stage)

        checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'last', 'last_checkpoint.zip')
        best_checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'best', 'best_checkpoint.zip')

        cm = CheckpointsManager(fsm).subscribe2trainer(trainer)
        best_cm = CheckpointsManager(fsm, prefix='best')
        bsd = BestStateDetector(trainer).subscribe2stage(stage).add_rule(lambda: np.mean(stage.get_losses()))
        events_container.event(bsd, 'BEST_STATE_ACHIEVED').add_callback(lambda b: best_cm.save_trainer_state(trainer))

        trainer.train()

        self.assertTrue(os.path.exists(best_checkpoint_file))
        best_cm.load_trainer_state(trainer)
        self.assertEqual(2, trainer.cur_epoch_id() - 1)

        self.assertTrue(os.path.exists(checkpoint_file))
        cm.load_trainer_state(trainer)
        self.assertEqual(4, trainer.cur_epoch_id() - 1)

    def test_events(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        stage = TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]))
        trainer = Trainer(BaseTrainConfig(model, [stage], SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(3)

        metrics_processor = MetricsProcessor().subscribe_to_stage(stage)
        metrics_processor.add_metric(DummyMetric())

        with MonitorHub(trainer) as mh:
            def on_epoch_start(local_trainer: Trainer):
                self.assertIs(local_trainer, trainer)

            def on_epoch_end(local_trainer: Trainer):
                self.assertIs(local_trainer, trainer)
                self.assertIsNone(local_trainer.train_config().stages()[0].get_losses())

            def stage_on_epoch_end(local_stage: TrainStage):
                self.assertIs(local_stage, stage)
                self.assertEqual(20, local_stage.get_losses().size)

            mh.subscribe2metrics_processor(metrics_processor)

            events_container.event(stage, 'EPOCH_END').add_callback(stage_on_epoch_end)
            events_container.event(trainer, 'EPOCH_START').add_callback(on_epoch_start)
            events_container.event(trainer, 'EPOCH_END').add_callback(on_epoch_end)

            trainer.train()

            self.assertEqual(None, trainer.train_config().stages()[0].get_losses())

    def test_metric_calc_in_train_loop(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))])),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]))]
        trainer = Trainer(BaseTrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm) \
            .set_epoch_num(2)

        mp = MetricsProcessor()
        mp.add_metrics_group(MetricsGroup('grp1').add(SimpleMetric()))
        mp.add_metrics_group(MetricsGroup('grp2').add(SimpleMetric()))

        mp.subscribe_to_stage(stages[0]).subscribe_to_stage(stages[1])
        mp.subscribe_to_trainer(trainer)

        trainer.train()
