import unittest
from random import seed
from random import randint
from dataset import AdsDataset
from text_rcnn_train import get_transform


class DatasetTest(unittest.TestCase):
    """
    Test Suite for dataset class.
    """
    def test_creat_dataset(self):
        # test if class can be instantiated
        AdsDataset()
        # test if class can be instantiated with transform function supplied
        AdsDataset(transforms=get_transform(train=True))

    def test_dataset_getitem(self):
        # test the get item function with random index
        dataset = AdsDataset(transforms=get_transform(train=True))
        seed(1)
        rand_index = randint(0, len(dataset) - 1)
        dataset[rand_index]
        # test the dimension of the item
        image, descriptor, target = dataset[-1]
        self.assertEqual(image.size(0), 3)
        self.assertEqual(image.size(1), image.size(2))
        self.assertEqual(descriptor.size(0), 300)
        self.assertEqual(target['boxes'].size(1), 4)
        self.assertEqual(target['boxes'].size(0), target['labels'].size(0))

    def test_dataset_len(self):
        # test the length function
        dataset = AdsDataset(transforms=get_transform(train=True))
        self.assertEqual(len(dataset), len(dataset.image_path))


if __name__ == "__main__":
    # Create the test suite from the cases above.
    dataset_testsuite = unittest.TestLoader().loadTestsFromTestCase(DatasetTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(dataset_testsuite)    