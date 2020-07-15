import torch
import numpy as np

from piepline.train_config.stages import TrainStage, ValidationStage
from piepline.train_config.train_config import BaseTrainConfig
from piepline.utils.fsm import FileStructManager
from piepline.predict import Predictor
from piepline.train import Trainer
from piepline.utils.checkpoints_manager import CheckpointsManager

from tests.common import UseFileStructure
from tests.data_processor_test import SimpleModel, SimpleLoss
from tests.data_producer_test import TestDataProducer


class PredictTest(UseFileStructure):
    def test_predict(self):
        test_data = {'data': torch.rand(1, 3)}

        model = SimpleModel()
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        cm = CheckpointsManager(fsm)

        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))])),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]))]
        trainer = Trainer(BaseTrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm)\
            .set_epoch_num(1)
        cm.subscribe2trainer(trainer)
        trainer.train()
        real_predict = trainer.data_processor().predict(test_data, is_train=False)

        fsm = FileStructManager(base_dir=self.base_dir, is_continue=True)
        cm = CheckpointsManager(fsm)

        predict = Predictor(model, checkpoints_manager=cm).predict(test_data)

        self.assertTrue(np.equal(real_predict.cpu().detach().numpy(), predict.cpu().detach().numpy()))
