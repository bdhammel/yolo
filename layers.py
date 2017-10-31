import tensorflow as tf
import numpy as np

from utils import lrelu

class ConvolutionalLayer:
    """A Convolutional Layer

    TODO
    ----
    Add in Dropout
    """

    ACTIVATION = {
            "leaky":lrelu,
            "relu":tf.nn.relu,
            "sigmoid":tf.nn.sigmoid,
            }

    def __init__(self, name, input_shape, shape, stride, pad, batch_normalize, activation, *args, **kwargs):
        """
        Args
        ----
        name (str) : name of layer, will be used as the namescope in the TF graph
        output_shape ([ints]) : the shape of image at the output
        shape ([int]) : a list of 4 ints, corresponds to the shape of the weights
        stride (int) : the window stride
        pad (str) : Padding type to use
        batch_normalize (bool) : run batch normalization after layer?
        activation (str) : the activation function to use, must correspond to key in ACTIVATION
        stride
        """
        self.name = name
        self.input_shape = input_shape
        self.fn = self.ACTIVATION[activation]
        self.strides = [1, stride, stride, 1]
        self.padding = "SAME"
        self.apply_batch_normalization = batch_normalize
        self.shape = shape
        
        # initialize weights. No need for a bias, batch norm takes care of that
        self.W = self._weight_variable(shape)

        # Initialize variables for batch normalization 
        zeros = np.zeros(self.get_output_shape()).astype(np.float32)
        self.running_mean = tf.Variable(
                zeros, 
                dtype=tf.float32, 
                trainable=False, 
                name="running_mean")
        self.running_var = tf.Variable(
                zeros, 
                dtype=tf.float32, 
                trainable=False, 
                name="running_var")

        gamma = np.ones(self.get_output_shape()).astype(np.float32)
        beta = zeros
        self.gamma = tf.Variable(gamma, dtype=tf.float32, name="beta")
        self.beta = tf.Variable(beta, dtype=tf.float32, name="gamma")

    @staticmethod
    def _weight_variable(shape, stddev=.1):
        """Initialize new weights, used for training the network from scratch

        Args
        ----
        shape (tuple, ints) : the shape of the weight tensor to initialize
        stddev (float) : standard deviation of the randomly initialized weights
        """
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(init, name="W")

    def _load_weights(self, file_name):
        """This method will be responsible for loading in pretrained weights
        """
        pass

    def _normalize(self, a, istraining, decay=1e-3):
        """Apply batch normalization

        Args
        ----
        a (Tensor) : imput thensor to normalize
        istraining (bool) : optional, denotes if the network is being trained, important
            for batch normalization 
        decay (float) : 

        Returns
        -------
        Tensor
        """
        if istraining:
            batch_mean, batch_var = tf.nn.moments(a, [0])
            update_rn_mean = tf.assign(
                    self.running_mean,
                    self.running_mean * decay + batch_mean * (1. - decay)
                    )
            update_rn_var = tf.assign(
                    self.running_var,
                    self.running_var * decay + batch_var * (1 - decay)
                    )
            with tf.control_dependencies([update_rn_mean, update_rn_var]):
                h = tf.nn.batch_normalization(
                        a, 
                        batch_mean, 
                        batch_var, 
                        self.beta, 
                        self.gamma, 
                        1e-4)
        else:
            h = tf.nn.batch_normalization(
                    a, 
                    self.running_mean, 
                    self.running_var, 
                    self.beta, 
                    self.gamma, 
                    1e-4)

        return h


    def forward(self, Z, istraining=False):
        """Pass data through layer and perform batch normalization if requested

        Args
        ----
        Z (Tensor) : input tensor into the convolutional layer
        istraining (bool) : optional, denotes if the network is being trained, important
            for batch normalization 

        Returns
        -------
        tensorflow Tensor
        """
        a = tf.nn.conv2d(
                Z, 
                self.W, 
                strides=self.strides, 
                padding=self.padding
                ) 

        if self.apply_batch_normalization:
            h = self._normalize(a, istraining)
        else:
            h = a

        return self.fn(h)

    def get_output_shape(self):
        """Return the shape at the output of the layer
        As a single strided convolutional layer, the shape will be the input
        size, but the depth will change depending on the filter depth

        Returns
        -------
        np.array : [batch_size, i, j, depth]
        """
        return np.array([*self.input_shape[:-1], self.shape[-1]])


class PoolingLayer:

    def __init__(self, input_shape, size, stride, **kwargs):
        """
        Args
        ----
        size (int) : size of the window to pool
        stride (int) : stride of the pooling window 
        """
        self.input_shape = input_shape
        self.strides = [1, stride, stride, 1]
        self.ksize = [1, size, size, 1]

    def forward(self, Z, *args):
        """Perform a pooling operation on the input data

        Args
        ----
        Z (Tensor) : input tensor into the convolutional layer
        *args : Because 'forward' is called on an ambiguous 'layer' in the network
            it could refer to a pooling layer or convolutional layer. *args 
            catches extra arguments which are not applicable to pooling.forward

        Returns
        -------
        tensorflow Tensor
        """
        return tf.nn.max_pool(
                   Z, 
                   ksize=self.ksize, 
                   strides=self.strides, 
                   padding='SAME'
                   )

    def get_output_shape(self):
        """Return the shape at the output of the layer
        As a pooling layer, the depth will remain unchanged, but the input shape
        will change based on the stride number

        Returns
        -------
        np.array : [batch_size, i, j, depth]
        """
        output_shape = np.ceil(
                np.divide(self.input_shape[:-1], self.strides[1:-1])
                )
        return np.array([*output_shape, self.input_shape[-1]]).astype(np.int32)



