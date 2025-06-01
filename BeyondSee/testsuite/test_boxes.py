import unittest
from unittest.mock import patch
import sys
sys.path.append('../')
from preprocess import boxes

class BoxesTest(unittest.TestCase):
    """
    Test Suite for boxes.
    """
    def test_preprocess_boxes(self):
        with patch('preprocess.boxes.load_symbols_annotation') as mocked_load_annot_method, \
                patch('preprocess.boxes.write_dict_to_json') as mocked_write_method:
            annot_fname = '../data/annotations/Symbols.json'
            mocked_load_annot_method.return_value = {
                'img1': [[1, 1, 1, 2, 'nature']],   # case 1: xmin equals xmax
                'img2': [[6, 2, 3, 3, 'happy/exciting']],   # case 2: xmin greater than xmax
                'img3': [[4, 4, 5, 4, 'sad'], [1, 1, 2, 2, 'angry']],   # case 3: ymin equals ymax, case 4: all coordinates within boundary
                'img4': [[4, 8, 5, 5, 'nature/natural']]   # case 5: ymin greater than ymax
            }
            boxes.preprocess_boxes(annot_fname)
            # test the methods are called with the corresponding file name
            mocked_load_annot_method.assert_called_once_with(annot_fname)
            # check the symbols annotation are preprocessed
            final_symbols = {
                'img2': [[3, 2, 6, 3, 'happy/exciting']],
                'img3': [[1, 1, 2, 2, 'angry']],
                'img4': [[4, 5, 5, 8, 'nature/natural']]
            }
            mocked_write_method.assert_called_once_with(annot_fname, final_symbols)
            # tes the boundary values
            mocked_load_annot_method.return_value = {
                'img1': [[-1, 1, 502, 2, 'nature']],
                'img2': [[0, 2, 4, 501, 'happy/exciting']]
            }
            boxes.preprocess_boxes(annot_fname)
            final_symbols = {
                'img2': [[0, 2, 4, 501, 'happy/exciting']]
            }
            mocked_write_method.assert_called_with(annot_fname, final_symbols)


if __name__ == "__main__":
    # Create the test suite from the cases above.
    boxes_testsuite = unittest.TestLoader().loadTestsFromTestCase(BoxesTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(boxes_testsuite)