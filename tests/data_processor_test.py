import shutil
import unittest

import torch
import numpy as np

from piepline.data_processor.data_processor import DataProcessor, TrainDataProcessor
from piepline.utils.fsm import FileStructManager
from piepline.utils.utils import dict_pair_recursive_bypass
from piepline.utils.checkpoints_manager import CheckpointsManager
from piepline.train_config.train_config import BaseTrainConfig

from tests.common import UseFileStructure, data_remove

__all__ = ['DataProcessorTest', 'TrainDataProcessorTest']


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

    @staticmethod
    def dummy_input():
        return torch.rand(3)


def compare_two_models(unittest_obj: unittest.TestCase, model1: torch.nn.Module, model2: torch.nn.Module):
    def on_node(n1, n2):
        if n1.device == torch.device('cuda:0'):
            n1 = n1.to('cpu')
        if n2.device == torch.device('cuda:0'):
            n2 = n2.to('cpu')
        unittest_obj.assertTrue(np.array_equal(n1.numpy(), n2.numpy()))

    state_dict1 = model1.state_dict().copy()
    state_dict2 = model2.state_dict().copy()

    dict_pair_recursive_bypass(state_dict1, state_dict2, on_node)


class NonStandardIOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        res1 = self.fc(x['data1'])
        res2 = self.fc(x['data2'])
        return {'res1': res1, 'res2': res2}


class DataProcessorTest(UseFileStructure):
    def test_initialisation(self):
        try:
            DataProcessor(model=SimpleModel())
        except:
            self.fail('DataProcessor initialisation raises exception')

    def test_prediction_output(self):
        model = SimpleModel()
        dp = DataProcessor(model=model)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertIs(type(res), torch.Tensor)

        model = NonStandardIOModel()
        dp = DataProcessor(model=model)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': {'data1': torch.rand(1, 3), 'data2': torch.rand(1, 3)}})
        self.assertIs(type(res), dict)
        self.assertIn('res1', res)
        self.assertIs(type(res['res1']), torch.Tensor)
        self.assertIn('res2', res)
        self.assertIs(type(res['res2']), torch.Tensor)

    def test_predict(self):
        model = SimpleModel().train()
        dp = DataProcessor(model=model)
        self.assertFalse(model.fc.weight.is_cuda)
        self.assertTrue(model.training)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertFalse(model.training)
        self.assertFalse(res.requires_grad)
        self.assertIsNone(res.grad)


class SimpleLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.tensor([1.], requires_grad=True)
        self.res = None

    def forward(self, predict, target):
        self.res = self.module * (predict - target)
        return self.res


class TrainDataProcessorTest(UseFileStructure):
    def test_initialisation(self):
        model = SimpleModel()
        train_config = BaseTrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        try:
            TrainDataProcessor(train_config=train_config)
        except:
            self.fail('DataProcessor initialisation raises exception')

    def test_prediction_train_output(self):
        model = SimpleModel()
        train_config = BaseTrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': torch.rand(1, 3)}, is_train=True)
        self.assertIs(type(res), torch.Tensor)

        model = NonStandardIOModel()
        train_config = BaseTrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': {'data1': torch.rand(1, 3), 'data2': torch.rand(1, 3)}}, is_train=True)
        self.assertIs(type(res), dict)
        self.assertIn('res1', res)
        self.assertIs(type(res['res1']), torch.Tensor)
        self.assertIn('res2', res)
        self.assertIs(type(res['res2']), torch.Tensor)

        self.assertTrue(model.training)
        self.assertTrue(res['res1'].requires_grad)
        self.assertIsNone(res['res1'].grad)
        self.assertTrue(res['res2'].requires_grad)
        self.assertIsNone(res['res2'].grad)

    def test_prediction_notrain_output(self):
        model = SimpleModel()
        train_config = BaseTrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': torch.rand(1, 3)}, is_train=False)
        self.assertIs(type(res), torch.Tensor)

        model = NonStandardIOModel()
        train_config = BaseTrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': {'data1': torch.rand(1, 3), 'data2': torch.rand(1, 3)}}, is_train=False)
        self.assertIs(type(res), dict)
        self.assertIn('res1', res)
        self.assertIs(type(res['res1']), torch.Tensor)
        self.assertIn('res2', res)
        self.assertIs(type(res['res2']), torch.Tensor)

        self.assertFalse(model.training)
        self.assertFalse(res['res1'].requires_grad)
        self.assertIsNone(res['res1'].grad)
        self.assertFalse(res['res2'].requires_grad)
        self.assertIsNone(res['res2'].grad)

    def test_predict(self):
        model = SimpleModel().train()
        train_config = BaseTrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        self.assertTrue(model.training)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertFalse(model.training)
        self.assertFalse(res.requires_grad)
        self.assertIsNone(res.grad)

    def test_train(self):
        model = SimpleModel().train()
        train_config = BaseTrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)

        self.assertFalse(model.fc.weight.is_cuda)
        self.assertTrue(model.training)
        res = dp.predict({'data': torch.rand(1, 3)}, is_train=True)
        self.assertTrue(model.training)
        self.assertTrue(res.requires_grad)
        self.assertIsNone(res.grad)

        with self.assertRaises(NotImplementedError):
            dp.process_batch({'data': torch.rand(1, 3), 'target': torch.rand(1)}, is_train=True)

        loss = SimpleLoss()
        train_config = BaseTrainConfig(model, [], loss, torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        res, out, target = dp.process_batch({'data': torch.rand(1, 3), 'target': torch.rand(1)}, is_train=True)
        self.assertTrue(model.training)
        self.assertTrue(loss.module.requires_grad)
        self.assertIsNotNone(loss.module.grad)
        self.assertTrue(np.array_equal(res.detach().cpu().numpy(), loss.res.data.numpy()))

    @data_remove
    def test_continue_from_checkpoint(self):
        def on_node(n1, n2):
            self.assertTrue(np.array_equal(n1.numpy(), n2.numpy()))

        model = SimpleModel().train()
        loss = SimpleLoss()

        for optim in [torch.optim.SGD(model.parameters(), lr=0.1), torch.optim.Adam(model.parameters(), lr=0.1)]:
            train_config = BaseTrainConfig(model, [], loss, optim)

            dp_before = TrainDataProcessor(train_config=train_config)
            before_state_dict = model.state_dict().copy()
            dp_before.update_lr(0.023)

            try:
                fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
                dp_before.save_state(CheckpointsManager(fsm).optimizer_state_file())
            except:
                self.fail("Exception on saving state when 'CheckpointsManager' specified")

            fsm = FileStructManager(base_dir=self.base_dir, is_continue=True)
            dp_after = TrainDataProcessor(train_config=train_config)
            try:
                cm = CheckpointsManager(fsm)
                cm.load_data_processor(dp_after)
            except:
                self.fail('DataProcessor initialisation raises exception')

            after_state_dict = model.state_dict().copy()

            dict_pair_recursive_bypass(before_state_dict, after_state_dict, on_node)
            self.assertEqual(dp_before.get_lr(), dp_after.get_lr())

            shutil.rmtree(self.base_dir)
