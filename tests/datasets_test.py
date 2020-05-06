import unittest

from piepline import BasicDataset


class TestingBasicDataset(BasicDataset):
    def _interpret_item(self, item) -> any:
        return self._items[item]


class BasicDatasetTest(unittest.TestCase):
    def test_init(self):
        try:
            TestingBasicDataset(list(range(12)))
            TestingBasicDataset([{'a': i, 'b': i * 2} for i in range(12)])
        except Exception as err:
            self.fail("Basic initialisation failed with error: ['{}']".format(err))

    def test_get_items_test(self):
        items = list(range(13))
        dataset = TestingBasicDataset(items)
        self.assertEqual(dataset.get_items(), items)
