import unittest
from unittest.mock import mock_open, patch
import os
import sys
sys.path.append('../')
from preprocess import labels

class LabelsTest(unittest.TestCase):
    """
    Test Suite for labels.
    """
    @patch('preprocess.labels.json')
    def test_get_symbol_cluster_name(self, mocked_json):
        # test the json is called to load the file
        filename = '../preprocess/clustered_symbol_list.json'
        labels.get_symbol_cluster_name(filename)
        mocked_json.loads.assert_called_once()
    
    @patch('preprocess.labels.json')
    def test_load_symbol_cluster(self, mocked_json):
        # test the json is called to load the file
        filename = '../preprocess/clustered_symbol_list.json'
        word_to_id, id_to_symbol = labels.load_symbol_cluster(filename)
        mocked_json.loads.assert_called_once()
        # test the return values are functioning
        for k, v in word_to_id.items():
            self.assertEqual(id_to_symbol[v], k)
        for k, v in id_to_symbol.items():
            self.assertEqual(word_to_id[v], k)

    @patch('preprocess.labels.json')
    def test_load_symbols_annotation(self, mocked_json):
        # test the json is called to load the file
        filename = '../data/annotations/Symbols.json'
        labels.load_symbols_annotation(filename)
        mocked_json.load.assert_called_once()

    @patch('preprocess.labels.load_symbols_annotation')
    def test_build_label_encoder(self, mocked_load_method):
        # test the load method is called
        le_path = os.path.join('..', labels.le_path)
        filename = '../data/annotations/Symbols.json'
        labels.build_label_encoder(filename, le_path)
        mocked_load_method.assert_called_once_with(filename)
        # test the file is written
        m = mock_open()
        with patch('builtins.open', m) as mocked_open, patch('pickle.dumps') as mocked_dumps:
            labels.build_label_encoder(filename, le_path)
            m.assert_called_with(le_path, 'wb')
            file = mocked_open()
            file.write.assert_called_once()
            mocked_dumps.assert_called_once()

    def test_preprocess_labels(self):
        with patch('preprocess.labels.load_symbols_annotation') as mocked_load_annot_method, \
            patch('preprocess.labels.load_symbol_cluster') as mocked_load_cluster_method, \
                patch('preprocess.labels.write_dict_to_json') as mocked_write_method:
            annot_fname = '../data/annotations/Symbols.json'
            cluster_fname = '../preprocess/clustered_symbol_list.json'
            mocked_load_annot_method.return_value = {
                'img1': [[1, 1, 2, 2, 'nature']],   # case 1: one label, label not in dict
                'img2': [[2, 2, 3, 3, 'happy/exciting']],   # case 2: more than one label, label in dict
                'img3': [[4, 4, 5, 5, 'sad'], [1, 1, 2, 2, 'angry']],   # case 3: one label, label in dict
                'img4': [[4, 4, 5, 5, 'nature/natural']],   # case 4: more than one label, label not in dict
            }
            word_to_id = {
                'happy': 1, 'sad': 2, 'angry': 3
            }
            id_to_word = {
                1: 'happy', 2: 'sad', 3: 'angry'
            }
            mocked_load_cluster_method.return_value = (word_to_id, id_to_word)
            labels.preprocess_labels(annot_fname, cluster_fname)
            # test the methods are called with the corresponding file name
            mocked_load_annot_method.assert_called_once_with(annot_fname)
            mocked_load_cluster_method.assert_called_once_with(cluster_fname)
            # check the symbols annotation are preprocessed
            final_symbols = {
                'img1': [[1, 1, 2, 2, 'nature']],
                'img2': [[2, 2, 3, 3, 'happy']],
                'img3': [[4, 4, 5, 5, 'sad'], [1, 1, 2, 2, 'angry']],
                'img4': [[4, 4, 5, 5, 'nature']],
            }
            mocked_write_method.assert_called_once_with(annot_fname, final_symbols)


if __name__ == "__main__":
    # Create the test suite from the cases above.
    labels_testsuite = unittest.TestLoader().loadTestsFromTestCase(LabelsTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(labels_testsuite)