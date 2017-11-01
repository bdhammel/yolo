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
import network

class ConcatinationAndSlicingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Start interactive session"""
        cls.session = tf.InteractiveSession()


    @classmethod
    def tearDownClass(cls):
        """Stop interactive session"""
        cls.session.close()


    def test_multi_dim_concatination(self):
        """Check that multi dimension arrays are being concatenated how I think 
        they are

        Check that the numbers along a given axis are concatenated, and that 
        after a slice along that axis I can regain the initial components
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

        Make sure that tensors don't need to be the same shape (in the 
        concatenation direction)
        """
        a = [ [1,2,3], [4,5,6], [7,8,9] ]
        b = [ [10,20], [40,50], [70,80] ]

        c = np.concatenate((a,b), axis=1)

        a_hat = c[:,:3]
        b_hat = c[:,3:]

        nptest.assert_array_equal(a, a_hat)
        nptest.assert_array_equal(b, b_hat)


    def test_numpy_multi_slice(self):
        """Check that using a list to slice numpy array concatenates them 
        correctly

        This is used in checking if the list_slice tensor method I created
        works
        """
        z = np.random.random((3,7,7,12))
        x = np.concatenate([z[...,[0]], z[...,[5]]], 3)
        x_hat = z[...,[0,5]]

        nptest.assert_array_equal(x, x_hat)

    @unittest.skip("Build this test")
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


    @unittest.skip("This is testing a tile slice, need to test tile")
    def test_tile_slice(self):
        """
        """
        a = [ [1,2,3], [4,5,6], [7,8,9] ]
        tfa = tf.constant(a)

        aa = np.tile(a, 2)

        tfaa = utils.tile_slice(tfa, index=1, axis=1, number=2)
        aa_hat = tfaa.eval()

        nptest.assert_array_equal(aa, aa_hat)

    def test_argmax(self):
        a = np.zeros(shape=(1,7,7,2))
        a[0,1,2,0] = 1
        a[0,5,3,1] = 1
        tfa = tf.constant(a)

        maxi = tf.argmax(a)



class LossFnTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = tf.InteractiveSession()
        cls.tinyyolo = network.YOLO(
                config_file="network_configs/tiny_yolo.json", debug=True)
        cls.tinyyolo.set_session(cls.session)


    @classmethod
    def tearDownClass(cls):
        cls.session.close()


    def test_x_sqr_err(self):
        """Check that the squared error for the x location is 
        generated correctly
        """
        # location at where the object "will be"
        i = j = 2

        # set the "actual" location and the "predicted" location 
        gnd_tru = np.zeros(shape=(1, 7, 7, 7))
        pred = np.zeros(shape=(1, 7, 7, 12))
        gnd_tru[:,i,j,[1]] = 3
        pred[:,i,j,[0,5]] = 5

        # define what the expected error should be 
        xerr = np.zeros(shape=(1,7,7,2))
        xerr[:,i,j,:] = 4 # error of 2, squared, at both of the bounding boxes

        # get the tensorflow calculated squared error
        tf_gnd_tru = tf.constant(gnd_tru, dtype=np.float32)
        tf_pred = tf.constant(pred, dtype=np.float32)
        self.tinyyolo._build_loss_fn(tf_gnd_tru, tf_pred)
        tf_xerr = self.tinyyolo.debug_dump['xerr']
        xerr_hat = tf_xerr.eval()

        nptest.assert_array_equal(xerr, xerr_hat)
        

    def test_x_sqr_err_reduced(self):
        """Check that the error reduces to the expected number

        Arrays are 0s except for the location of the object
        """
        i = j = 2

        gnd_tru = np.zeros(shape=(1, 7, 7, 7))
        pred = np.zeros(shape=(1, 7, 7, 12))

        gnd_tru[:,i,j,[1]] = 3
        pred[:,i,j,[0,5]] = 5

        tf_gnd_tru = tf.constant(gnd_tru, dtype=np.float32)
        tf_pred = tf.constant(pred, dtype=np.float32)
        self.tinyyolo._build_loss_fn(tf_gnd_tru, tf_pred)
        tf_xerr = tf.reduce_sum(self.tinyyolo.debug_dump['xerr'])
        xerr_hat = tf_xerr.eval()
        self.assertEqual(4+4, xerr_hat)

    def test_indicator_obj_ij_correctly_made(self):
        """confirm that the indicator ij matrix is made correctly """
        pass


    def test_x_sqr_err_times_indicator(self):
        """Check that the squared error does not include error from cells without
        and object

        During operation, yolo will predict a coor value (x,y,w,h) for every S
        make sure that these do not add to the error, unless there is an object 
        at S
        """
        # S location 
        i = j = 2

        # set the "actual" location and the "predicted" location 
        # initialize the predicted location with random numbers
        gnd_tru = np.zeros(shape=(1, 7, 7, 7))
        pred = np.random.random(size=(1, 7, 7, 12))
        gnd_tru[:,i,j,0] = 1    # set obj location in indicator matrix
        gnd_tru[:,i,j,1] = 3    # set tru obj x coor
        pred[:,i,j,[0,5]] = 5   # set "predicted" obj coor define what the expected error should be 
        reduced_xerr = 4 # error of 2, squared, at both of the bounding boxes

        # get the tensorflow calculated squared error
        tf_gnd_tru = tf.constant(gnd_tru, dtype=np.float32)
        tf_pred = tf.constant(pred, dtype=np.float32)
        self.tinyyolo._build_loss_fn(tf_gnd_tru, tf_pred)
        indicator = self.tinyyolo.debug_dump['indicator_obj_ij']
        tf_xerr = indicator * self.tinyyolo.debug_dump['xerr']
        xerr_reduced_hat = tf.reduce_sum(tf_xerr).eval()

        nptest.assert_array_equal(reduced_xerr, xerr_reduced_hat)

    @unittest.skip("Build this test")
    def test_confedece_error_is_correct(self):
        """Make sure the network correctly generates the confidence matrix 
        for the ground truth.

        This matrix should be shape batch_sz x 7 x 7 x 2 with 1's at the S location
        of the object for both bounding boxes
        """
        # S location 
        i = j = 2

        # set the "actual" location and the "predicted" location 
        # initialize the predicted location with random numbers
        gnd_tru = np.zeros(shape=(1, 7, 7, 7))
        pred = np.random.random(size=(1, 7, 7, 12))
        gnd_tru[:,i,j,0] = 1    # set obj location in indicator matrix
        gnd_tru[:,i,j,1] = 3    # set tru obj x coor
        pred[:,i,j,[0,5]] = 5   # set "predicted" obj coor


    def test_indicator_noobj_is_opposite_of_indicator_obj(self):
        """ Make sure the the indicator and the indicator_noobj arrays
        are opposite from one another

        The arrays should differ by 1 at every element in the array
        """
        # S location 
        i = j = 2

        # set the "actual" location and the "predicted" location 
        # initialize the predicted location with random numbers
        gnd_tru = np.zeros(shape=(1, 7, 7, 7))
        pred = np.zeros(shape=(1,7,7,12))
        gnd_tru[:,i,j,0] = 1    # set obj location in indicator matrix
        gnd_tru[:,i,j,1] = 3    # set tru obj x coor
        pred[...,[0,5]] = np.random.random(size=(1, 7, 7, 2))


        # the tensorflow calculates loss
        tf_gnd_tru = tf.constant(gnd_tru, dtype=np.float32)
        tf_pred = tf.constant(pred, dtype=np.float32)
        self.tinyyolo._build_loss_fn(tf_gnd_tru, tf_pred)

        indicator_obj = self.tinyyolo.debug_dump['indicator_obj_ij']
        indicator_noobj = self.tinyyolo.debug_dump['indicator_noobj_ij']

        # asset that the arrays are different by 1 at every element
        tf_indicator_diff =  tf.abs(indicator_obj - indicator_noobj)
        indicator_diff = tf_indicator_diff.eval().sum()

        expected_diff = 7 * 7 * 2

        self.assertEqual(expected_diff, indicator_diff)


    @unittest.skip("Need to confirm indicator_obj_ij before doing this test")
    def test_loss_is_being_corectly_calculate_for_x_err(self):
        """Check that the loss function is being correctly calculated for 
        and error in x
        """
        # S location 
        i = j = 2

        # set the "actual" location and the "predicted" location 
        # initialize the predicted location with random numbers
        gnd_tru = np.zeros(shape=(1, 7, 7, 7))
        pred = np.zeros(shape=(1,7,7,12))
        gnd_tru[:,i,j,0] = 1    # set obj location in indicator matrix
        gnd_tru[:,i,j,1] = 3    # set tru obj x coor
        pred[...,[0,5]] = np.random.random(size=(1, 7, 7, 2))

        # the tensorflow calculates loss
        tf_gnd_tru = tf.constant(gnd_tru, dtype=np.float32)
        tf_pred = tf.constant(pred, dtype=np.float32)
        tf_loss_hat = self.tinyyolo._build_loss_fn(tf_gnd_tru, tf_pred)
        loss_hat = tf_loss_hat.eval()

        # The loss for an error in x should be 
        cerr_obj = np.square(1 - pred[:,i,j,[4,9]]).sum()
        cerr_noobj = np.square(pred[...,[4,9]]).sum()
        xerr = np.square(pred[:,i,j,[0,5]]).sum()

        lambda_coor = 5.
        lambda_noobj = .5
        indicator_obj_ij = self.tinyyolo.debug_dump['indicator_obj_ij'].eval()
        indicator_noobj_ij = self.tinyyolo.debug_dump['indicator_noobj_ij'].eval()
        loss = lambda_coor * xerr + cerr_obj \
                + lambda_noobj * cerr_noobj

        # Sanity checks
        self.assertEqual(0, self.tinyyolo.debug_dump['yerr'].eval().sum())
        self.assertEqual(0, self.tinyyolo.debug_dump['werr'].eval().sum())
        self.assertEqual(0, self.tinyyolo.debug_dump['herr'].eval().sum())
        self.assertEqual(0, self.tinyyolo.debug_dump['perr'].eval().sum())

        # check loss value
        self.assertEqual(loss, loss_hat)


    def test_classification_error(self):
        pass


class PredictionTests(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        """Start interactive session"""
        cls.session = tf.InteractiveSession()


    @classmethod
    def tearDownClass(cls):
        """Stop interactive session"""
        cls.session.close()


    def test_yolo_works_as_classifier(self):
        pass





if __name__ == '__main__':
    """run tests
    """
    unittest.main()


