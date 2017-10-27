import tensorflow as tf
import unittest
from numpy import testing as nptest
import numpy as np
import os
import sys

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir_name not in sys.path:
    sys.path.append(parent_dir_name)

import utils

class ConcatinationAndSlicingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = tf.InteractiveSession()


    @classmethod
    def tearDownClass(cls):
        cls.session.close()


    def test_multi_dim_concatination(self):
        """check that multi dimension arrays are being concatenated how I think they 
        are
        """

        a = [ [1,2,3], [4,5,6], [7,8,9] ]
        b = [ [10,20,30], [40,50,60], [70,80,90] ]

        c = np.concatenate((a,b), axis=1)

        a_hat = c[:,:3]
        b_hat = c[:,3:]

        nptest.assert_array_equal(a, a_hat)
        nptest.assert_array_equal(b, b_hat)


    def test_off_dim_concatination(self):
        """Make sure that arrays with different dimensions in the concatenation
        direction can be concatenated
        """
        a = [ [1,2,3], [4,5,6], [7,8,9] ]
        b = [ [10,20], [40,50], [70,80] ]

        c = np.concatenate((a,b), axis=1)

        a_hat = c[:,:3]
        b_hat = c[:,3:]

        nptest.assert_array_equal(a, a_hat)
        nptest.assert_array_equal(b, b_hat)


    def test_i_am_using_tile_correctly(self):
        """Test that tile is concatenating arrays in the way that I expect
        """
        a = [ [1,2,3], [4,5,6], [7,8,9] ]

        aa = np.tile(a, 2)

        a_hat = aa[:,:3]

        nptest.assert_array_equal(a, a_hat)


    def test_numpy_multi_slice(self):
        """Check that using a list to slice numpy array concatenates them 
        correctly
        """
        z = np.random.random((3,7,7,12))
        x = np.concatenate([z[...,[0]], z[...,[5]]], 3)
        x_hat = z[...,[0,5]]

        nptest.assert_array_equal(x, x_hat)


    def test_slice_and_keep_shape_function(self):
        self.assertTrue(False)


    def test_list_slice_function(self):
        """Test that my list slice function operates the same way the a multi slice,
        or slicing with a list, works in numpy
        """
        z = np.random.random(size=(3, 7, 7, 12))
        x = z[...,[0,5]]
        tfz = tf.constant(z)
        tfx_hat = utils.list_slice(tfz, [0, 5], axis=3)
        x_hat = tfx_hat.eval()

        nptest.assert_array_equal(x, x_hat)


    def test_tile_slice(self):
        """
        """
        a = [ [1,2,3], [4,5,6], [7,8,9] ]
        tfa = tf.constant(a)

        aa = np.tile(a, 2)

        tfaa = utils.tile_slice(tfa, index=, axis=3, number=2)
        aa_hat = tfaa.eval()

        nptest.assert_array_equal(aa, aa_hat)

    def test_argmax(self):
        a = np.zeros(shape=(1,7,7,2))
        a[0,1,2,0] = 1
        a[0,5,3,1] = 1
        tfa = tf.constant(a)

        maxi = tf.argmax(a)




class NetworkTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    """run tests
    """
    unittest.main()


