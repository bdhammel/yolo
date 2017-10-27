import tensorflow as tf
import numpy as np
import json

import utils
from layers import ConvolutionalLayer, PoolingLayer
from trainer import YOLOTrainer

class YOLO:

    def __init__(self, config_file, weights_dir="tiny_yolo_weights/"):
        """Initialize the YOLO network

        Args
        ----
        config_file (str) : 
        weights_dir (str) : 
        """
        self.config = self._load_config(config_file)

        self._init_placeholders()
        self.layers = self._build_layers()

        self.pred = self.forward(self.yolo_in, istraining=True)
        self.loss = self._build_loss_fn() 
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)


    def _parse_yolo_out(self, output):
        """Extract relevant information from the output of the network

        Args
        ----
        output is batch_size x S x S x (B*5 + C)
        wherein the S x S corresponds to image dub divisions 
        and B*5 + C depth corresponds to network predictions
            - x, y, width, height for the bounding box's rectangle
            - the confidence score
            - probability distribution over C classes
        """
        pass


    def _perdict(self):
        """
        """
        pass


    def _build_layers(self):
        """Build the network of convolutional and pooling layers

        Build layers off config file

        Returns
        -------
        layers ([ConvolutionalLayer, PoolingLayers]): List of the layers in the 
            network in order
        """
        layers = [] # initialize list to save layers

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
        """
        
        Notes
        -----
        Only works with 2 bounding boxes!
        """

        shape = indicator_obj.get_shape().as_list()
        shape[shape==None] = -1

        stacked_indicator_obj = tf.concat([indicator_obj, indicator_obj], axis=3)
        _indicator_iou = tf.multiply(iou, stacked_indicator_obj)
        
        b1 = tf.reshape(
                tf.cast(
                    tf.argmax(_indicator_iou, axis=3), 
                    dtype=tf.float32),
                shape, 
                )

        ones = tf.ones_like(b1, dtype=tf.float32)

        b0 = tf.reshape(
                ones - b1, 
                shape
                )

        return tf.concat([b0, b1], axis=3)


    def _build_loss_fn(self):
        """Construct the loss function of the yolo network

        """
        lambda_noobj = 0.5  # scale cost for no object (weight less)
        lambda_coord = 5.0    # scale cost for the box coordinates (weight more)

        # Find the indices of the box dimension-values, depending on number of 
        # bounding boxes used
        B = 2
        j = np.array(range(B), dtype=np.int32)

        # Get the predicted values
        x_hat = utils.list_slice(self.pred, j+0, axis=3)
        y_hat = utils.list_slice(self.pred, j+1, axis=3)
        w_hat = utils.list_slice(self.pred, j+2, axis=3)
        h_hat = utils.list_slice(self.pred, j+3, axis=3)
        C_hat = utils.list_slice(self.pred, j+4, axis=3)
        p_hat = self.pred[...,5*B:]

        # Get the true values for the batch
        indicator_obj_i = utils.slice_and_keep_shape(self.gnd_tru, index=0, axis=3)
        x = utils.tile_slice(self.gnd_tru, index=1, axis=3, number=2)
        y = utils.tile_slice(self.gnd_tru, index=2, axis=3, number=2)
        w = utils.tile_slice(self.gnd_tru, index=3, axis=3, number=2)
        h = utils.tile_slice(self.gnd_tru, index=4, axis=3, number=2)
        p = self.gnd_tru[...,5:] 
        C = tf.concat([indicator_obj_i, indicator_obj_i], axis=3)

        ## Construct the loss function

        # Construct the indicator matrix for the most relevant predictions of 
        # confidence based on the IofU
        iou = utils.tf_iou(x, y, w, h, x_hat, y_hat, w_hat, h_hat)
        indicator_obj_ij = self._construct_indicator_coor(iou, indicator_obj_i)

        # Construct the indicator matrix for all prediction not associated with 
        # and object, same as indicator ^ 1, logical or  
        indicator_noobj_ij = 1 - indicator_obj_ij

        # get the errors for each predicted component 
        x_err = tf.square(x - x_hat)
        y_err = tf.square(y - y_hat)
        w_err = tf.square(tf.sqrt(w) - tf.sqrt(w_hat))
        h_err = tf.square(tf.sqrt(h) - tf.sqrt(h_hat))
        c_err = tf.square(C - C_hat)
        p_err = tf.square(p - p_hat)

        # loss function 
        loss = tf.reduce_sum(
                lambda_coord * tf.reduce_sum(
                    indicator_obj_ij * (
                        x_err + y_err
                        ),
                    axis=3
                    )
            )
        """
         + w_err + h_err + c_err
        + indicator_noobj_ij
        + lambda_noobj * tf.reduce_sum(
            indicator_noobj_ij * (
                c_err
                )
        + indicator_obj_i * tf.reduce_sum(
            p_err,
            axis=3,
            keep_dims=True
            )
        ) 
        """

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


    def train(self, trainer, epochs=135, batch_sz=64):

        for e in range(epochs):
            for batch_in, batch_tru in trainer.get_batches(batch_sz):
                l, _ = self.session.run(
                        [self.loss, self.train_op], 
                        feed_dict={
                            self.yolo_in:batch_in,
                            self.gnd_tru:batch_tru
                        })

                print("e: ", e, "loss: ", l)

        

if __name__ == "__main__":

    yolo_trainer = YOLOTrainer("./data/images.npy", "./data/labels.npy")

    tinyyolo = YOLO(config_file="network_configs/tiny_yolo.json")

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tinyyolo.set_session(session)
        tinyyolo.train(yolo_trainer)

