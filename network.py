"""TensorFlow implementation of YOLO 

Image localization and classification network


References
----------

(*) J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi, 
"You only look once: Unified, real-time object detection," 
CoRR, vol. abs/1506.02640, 2015.

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json

import utils
from layers import ConvolutionalLayer, PoolingLayer
from trainer import YOLOTrainer

class YOLO:

    def __init__(self, config_file, weights_dir="tiny_yolo_weights/", 
                    learning_rate=1e-2, debug=False, save_path=None):
        """Initialize the YOLO network based on a configuration file

        Note
        ----
        Currently does not support loading of weights

        Args
        ----
        config_file (str) :  path to the location of a json config file
        weights_dir (str) :  path to saved weights
        """
        self.debug = debug

        print("Building the YOLO network")
        self.config = self._load_config(config_file)

        self._init_placeholders()
        self.layers = self._build_layers()

        self.pred = self.forward(self.yolo_in, istraining=True)
        self.loss = self._build_loss_fn(self.gnd_tru, self.pred) 
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        print("...done")

        # initialize a saver to save the weights
        self.saver = tf.train.Saver()


    def _parse_yolo_out(self, output):
        """Extract relevant information from the output of the network

        output is batch_size x S x S x (B*5 + C)
        wherein the S x S corresponds to image dub divisions 
        and B*5 + C depth corresponds to network predictions
            - x, y, width, height for the bounding box's rectangle
            - the confidence score
            - probability distribution over C classes

        Args
        ----
        output (tensor) : the output tensor from the network 

        Returns
        ------
        """
        pass


    def perdict(self):
        """
        """
        pass


    def _build_layers(self):
        """Build the network of convolutional and pooling layers

        Iterate through the config file specifications and Build layers 

        Returns
        -------
        layers ([ConvolutionalLayer, PoolingLayers]): List of the layers in the 
            network, in order of propagation 
        """
        # initialize list to save layers
        layers = []

        # number of filters at the input (3 color channels)
        Fi = 3 
        
        # Shape of the input image
        input_shape = np.array(self.yolo_in.get_shape().as_list()[1:])

        # Iterate through the layers in the config file and build them into 
        # the network 
        for layer_config in self.config["layers"]:

            # Build a convolutional type layer
            if layer_config["type"] == "convolutional":
                S1 = S2 = layer_config["size"] # size of window 
                Fo = layer_config["filters"] # number of filters in layer

                # build layer
                layer = ConvolutionalLayer(
                            input_shape=input_shape,
                            shape=[S1, S2, Fi, Fo],
                            **layer_config
                            )

                # the output filter number will be the input filter number for 
                # next layer
                Fi = Fo

            # Build a pooling layer
            elif layer_config["type"] == "maxpool":
                # build layer
                layer = PoolingLayer(
                            input_shape=input_shape,
                            **layer_config
                            )

            # Doesn't support any other layers at this time, break the code if 
            # something different is given 
            else:
                print("somehting wrong")
                assert False

            # Get the shape of the data at the output of the last layer built
            input_shape = layer.get_output_shape()

            # Save the layer in the network 
            layers.append(layer)

        return layers


    def _init_placeholders(self):
        """initialize placeholders for the graph

        Attributes
        ----------
        self.in_ (tf placeholder) : input image. batch x Width x Height x color 
        self.true_ (tf placeholder) :
        """
        self.yolo_in = tf.placeholder(
                tf.float32, 
                [None, 208, 208, 3],
                name="input")

        self.gnd_tru = tf.placeholder(
                tf.float32, 
                [None, 7, 7, 7],
                name="true")


    def _load_config(self, config_file):
        """Load the network architecture from a JSON config file

        Args
        ----
        config_file (str) : path to the config file to be loaded

        Returns 
        -------
        python dictionary containing the network parameters to be loaded 
        """
        with open(config_file) as f:
            return json.load(f)

    def _construct_indicator_coor(self, iou, indicator_obj):
        """Find the bounding box with the best prediction

        For network with 2 bounding boxes, one object, and anchor boaxes 7x7
        the output will be batch_sz x 7 x 7 x 2. 
        All elements will be 0, except for one S, B location, which is the best
        prediction
        
        Notes
        -----
        Only works with 2 bounding boxes!

        Args
        ----
        iou (tensor) : the calculated intersection of union. 
            has dims batch_sz x S x S x B
        indicator_obj (tensor) :
            has dims batch_sz x S x S x B

        Returns
        -------
        Tensor : the tensor marking the location of the best representation of the
            bounding box of the predicted object
        """

        # save the shape so that the output can be remade to the same dimensions
        shape = indicator_obj.get_shape().as_list()
        shape[shape==None] = -1

        # stack two of the indicators for object location together 
        # one of these will be set to 0, the one that doesn't match the iou
        _stacked_indicator_obj = tf.concat([indicator_obj, indicator_obj], axis=3)

        # this only selects the iou at the object locations, sets iou every
        # where else to 0
        _indicator_iou = tf.multiply(iou, _stacked_indicator_obj)

        # mark all locations without an object as -1
        # and add it to the indicator with iou
        # this is done so that argmax wont select these
        # _stacked_indicator_noobj = -1 * (ones - _stacked_indicator_obj)
        # _indicator_iou = _indicator_iou + _stacked_indicator_noobj
        
        ## Find the bounding box with the best IOU 
        # This will find the location with the highest IOU
        # and return its index, 0 or 1, for 1st (0) or 2nd (1) bounding box
        b1 = tf.reshape(
                tf.cast(
                    tf.argmax(_indicator_iou, axis=3), 
                    dtype=tf.float32),
                shape, 
                )

        # fip all of the values from the b1 array
        # this is because argmax will return a 0 for the for the highest iou 
        # in the first bounding box, but we want to represent this location as a 
        # 1. However, doing this will also, set 1 to all the locations without
        # and object, so we need to multiply by the indicator matrix again
        ones = tf.ones_like(b1, dtype=tf.float32)
        b0 = indicator_obj * tf.reshape(
                ones - b1, 
                shape
                )

        return tf.concat([b0, b1], axis=3)


    def _build_loss_fn(self, gnd_tru, pred):
        """Construct the loss function of the yolo network

        Args
        ----
        gnd_tru (tensor) : a tensor representing the ground truth of a given 
            input image. Has dims batch_sx x S x S x (5 + C)
        pred (tensor) : the output tensor of the yolo network.
            has dims batch_sz x S x S x (B*5 + C)

        Returns
        -------
        tf_operation (float32) : the calculated loss for that batch 

        """
        lambda_noobj = 0.5  # scale cost for no object (weight less)
        lambda_coord = 5.0  # scale cost for the box coordinates (weight more)

        # Find the indices of the box dimension-values, depending on number of 
        # bounding boxes used
        _B = 2   # Hard code in number of boxes, this should be dynamic in the future
        _j = np.array(range(_B), dtype=np.int32)

        # Get the predicted values
        x_hat = utils.list_slice(pred, 5*_j+0, axis=3)
        y_hat = utils.list_slice(pred, 5*_j+1, axis=3)
        w_hat = utils.list_slice(pred, 5*_j+2, axis=3)
        h_hat = utils.list_slice(pred, 5*_j+3, axis=3)
        C_hat = utils.list_slice(pred, 5*_j+4, axis=3)
        p_hat = pred[...,5*_B:]

        # Get the true values for the batch
        indicator_obj_i = utils.slice_and_keep_dims(gnd_tru, index=0, axis=3)
        x = utils.tile_slice(gnd_tru, index=1, axis=3, number=2)
        y = utils.tile_slice(gnd_tru, index=2, axis=3, number=2)
        w = utils.tile_slice(gnd_tru, index=3, axis=3, number=2)
        h = utils.tile_slice(gnd_tru, index=4, axis=3, number=2)
        p = gnd_tru[...,5:] 
        C = tf.concat([indicator_obj_i, indicator_obj_i], axis=3)

        ## Construct the loss function

        # Construct the indicator matrix for the most relevant predictions of 
        # confidence based on the IofU
        _iou = utils.tf_iou(x, y, w, h, x_hat, y_hat, w_hat, h_hat)
        indicator_obj_ij = self._construct_indicator_coor(_iou, indicator_obj_i)

        # Construct the indicator matrix for all prediction not associated with 
        # and object, same as indicator ^ 1, logical or  
        indicator_noobj_ij = 1 - indicator_obj_ij

        # get the squared errors for each predicted component 
        x_err = tf.square(x - x_hat)
        y_err = tf.square(y - y_hat)
        w_err = tf.square(tf.sqrt(w) - tf.sqrt(w_hat))
        h_err = tf.square(tf.sqrt(h) - tf.sqrt(h_hat))
        c_err = tf.square(C - C_hat)
        p_err = tf.square(p - p_hat)

        # save squared error calculation for unit tests
        if self.debug:
            self.debug_dump = {
                    "xerr":x_err, "yerr":y_err, 
                    "werr":w_err, "herr":h_err,
                    "cerr":c_err, "perr":p_err,
                    "indicator_obj_i":indicator_obj_i,
                    "indicator_obj_ij":indicator_obj_ij,
                    "indicator_noobj_ij":indicator_noobj_ij,
                    }

        # loss function. Equation 3 from (*)
        loss = tf.reduce_sum(

                tf.reduce_sum(
                    indicator_obj_ij * (
                        lambda_coord * (x_err + y_err + w_err + h_err)
                        + c_err
                    ) 
                    + indicator_noobj_ij * lambda_noobj * (c_err),
                    axis=3,
                    keep_dims=True
                    )

                + indicator_obj_i * tf.reduce_sum(
                    p_err,
                    axis=3,
                    keep_dims=True
                    )

                )

        return loss
            

    def forward(self, Z, istraining=False):
        """Propagate an input image through the network

        Args
        ----
        Z (Tensor) : initial input image into the network
        training (bool) : optional, denotes if the network is being trained, 
            important for batch normalization 

        Returns
        -------
        Tensor : output layer of the network 
            of shape batch_sz x S x S x (5*B + P)
        """
        for layer in self.layers:
            Z = layer.forward(Z, istraining)

        return Z


    def set_session(self, session):
        """Set the session to be referenced later

        Args
        ----
        session (tf.Session()) : tensorflow session object
        """
        self.session = session


    def train(self, trainer, epochs=135, batch_sz=64, save_model=True, 
            save_path=None, show_fig=True):
        """

        Args
        ----
        trainer (YOLOTrainer) : a yolo trainer class with data loaded 
        epochs (int) : number of epochs to train over
        batch_sz (int) : batch size to use

        Specs from the paper, (*)
        -------------------------
        Epochs: 135
        Learning rate : 1e-2 - 1e-4
        """

        print("Training the network")

        costs = []

        for e in range(epochs):
            batch_count = 0
            for batch_in, batch_tru in trainer.get_batches(batch_sz):
                l, _ = self.session.run(
                        [self.loss, self.train_op], 
                        feed_dict={
                            self.yolo_in:batch_in,
                            self.gnd_tru:batch_tru
                        })

                print("\tepoch: ", e, "batch: ", batch_count, "loss: ", l)
                batch_count += 1
                costs.append(l)

                if self.debug:
                    try:
                        self.debug_output()
                    except:
                        pass
        
        # plot the network loss over time to see improvement 
        if show_fig:
            plt.plot(costs)
            plt.show()

        # Save the trained weights to the given path
        if save_model and save_path:
            print("Saving the trained weights")
            self.saver.save(sess=self.session, save_path=save_path)
            print("...done")


    def restore_saved_model(self, path=None):
        """Restore a saved model

        Args
        ----
        path (str) : the path to the directory containing the saved session
        """
        self.saver.restore(sess=self.session, save_path=self.save_path)


if __name__ == "__main__":

    yolo_trainer = YOLOTrainer("./data/images.npy", "./data/labels.npy")
    tinyyolo = YOLO(config_file="network_configs/tiny_yolo.json", debug=True)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tinyyolo.set_session(session)
        tinyyolo.train(yolo_trainer, save_path="saved_weights/tiny_yolo",
                save_model=True)




