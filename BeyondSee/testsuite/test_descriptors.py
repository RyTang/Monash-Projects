import os
import unittest
import sys
from unittest.mock import patch
import numpy as np
import torch
from torch.nn.modules.module import T
sys.path.append('../')
from preprocess import descriptors


class DescriptorsTest(unittest.TestCase):
    """
    Test Suite for descriptors.
    """
    @patch('preprocess.descriptors.api')
    def test_api(self, mock_api):
        # test api called once when model exists
        t = descriptors.TextEmbedModel()
        if not os.path.exists(t.embed_model_name + '.model'):
            mock_api.load.assert_called_once_with(t.embed_model)

    def test_sentiments_transform(self):
        s = descriptors.SentimentPreProcessor(root='../data/annotations')
        # case 1: all ids have the same frequency
        lst = [['1'], ['15'], ['30']]
        vec = s.transform(lst)
        np_array = np.array(s.model.get_vector(s.id_to_word[1]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 2: one id has the most frequency
        lst = [['1', '15'], ['15'], ['15', '30']]
        vec = s.transform(lst)
        np_array = np.array(s.model.get_vector(s.id_to_word[15]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 3: more than one id has the same frequency
        lst = [['1', '15', '30'], ['15', '30'], ['15', '30']]
        vec = s.transform(lst)
        np_array = np.array(s.model.get_vector(s.id_to_word[15]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))

    def test_topics_transform(self):
        t = descriptors.TopicsPreProcessor(root='../data/annotations')
         # case 1: all ids have the same frequency
        lst = ["28", "18", "39"]
        vec = t.transform(lst)
        np_array = np.array(t.text_embed_model.get_vector_rep(t.id_to_word[28]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 2: one id has the most frequency
        lst = ["28", "18", "28"]
        vec = t.transform(lst)
        np_array = np.array(t.text_embed_model.get_vector_rep(t.id_to_word[28]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 3: more than one id has the same frequency
        lst = ["28", "18", "28", "18", "39"]
        vec = t.transform(lst)
        np_array = np.array(t.text_embed_model.get_vector_rep(t.id_to_word[28]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 4: no id but text
        lst = ["nature", "wild life", "zoo"]
        vec = t.transform(lst)

    def test_strategies_transform(self):
        s = descriptors.StrategiesPreProcessor(root='../data/annotations')
        # case 1: all ids have the same frequency
        lst = [['1'], ['5'], ['10']]
        vec = s.transform(lst)
        np_array = np.array(s.text_embed_model.get_vector_rep(s.id_to_word[1]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 2: one id has the most frequency
        lst = [['1', '5'], ['5'], ['5', '10']]
        vec = s.transform(lst)
        np_array = np.array(s.text_embed_model.get_vector_rep(s.id_to_word[5]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 3: more than one id has the same frequency
        lst = [['1', '5', '10'], ['5', '10'], ['5', '10']]
        vec = s.transform(lst)
        np_array = np.array(s.text_embed_model.get_vector_rep(s.id_to_word[5]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 4: no id but text
        lst = [["scare tactics"], ["written message"], ["scary"]]
        vec = s.transform(lst)


if __name__ == "__main__":
    # Create the test suite from the cases above.
    labels_testsuite = unittest.TestLoader().loadTestsFromTestCase(DescriptorsTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(labels_testsuite)