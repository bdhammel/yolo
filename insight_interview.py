
from network import YOLO
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from trainer import YOLOTrainer
from tests import test_yolo

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{bbm}']


def visulize_yolo_input_data():
    """Demonstration of the data being used to train the YOLO network
    """

    # Make a helper class to assist with feeding data into the YOLO network 
    yolo_trainer = YOLOTrainer()

    # Save data into the yolo trainer 
    # data is a series of images with stop signs and yield signs
    # Images are all RGB and of different sizes and shapes
    # all images are cropped and resized to be 208 x 208 
    # save the data into a pickle file for faster loading in the future 
    image_dir = "./data/images"
    labels_dir = "./data/labels"
    yolo_trainer.clean_and_pickle(image_dir, labels_dir, show=True, save=False)


def run_yolo_network():
    """Feed data into the network 

    Show that the loss does not converge 

    Specs from the paper, (*)
    -------------------------
    Epochs: 135
    Learning rate : 1e-2 - 1e-4

    """

    # import data from binary file
    yolo_trainer = YOLOTrainer("./data/images.npy", "./data/labels.npy")

    # Initialize the network, show that decreasing learning rate doesn't 
    # improve convergence
    # learning_rate=1e-2
    # learning_rate=1e-8
    tinyyolo = YOLO(
            config_file="network_configs/tiny_yolo.json", 
            debug=True,
            learning_rate=1e-2
            )

    # train
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tinyyolo.set_session(session)
        tinyyolo.train(yolo_trainer, batch_sz=1)


def generate_latex_loss_eqn():
    """Display the YOLO Loss function using matplotlib
    """
    latex = r"$\lambda_{\rm coor} \sum\limits^{S^2}_i \sum\limits^{B}_j \mathbbm{1}_{ij}^{\rm obj}$"
    latex += r"$\left [ (x_i-\hat{x}_i)^2 + (y_i-\hat{y}_i)^2 + (\sqrt{w_i}-\sqrt{\hat{w}_i})^2 + (\sqrt{h_i}-\sqrt{\hat{h}_i})^2\right ]$" 
    latex += "\n"
    latex += r"$+ \sum\limits^{S^2}_i \sum\limits^B_j \mathbbm{1}^{\rm obj}_{ij} ( C_i - \hat{C}_i )^2$"
    latex += "\n"
    latex += r"$+ \lambda_{\rm noobj} \sum\limits^{S^2}_i \sum\limits^B_j \mathbbm{1}^{\rm noobj}_{ij} ( C_i - \hat{C}_i )^2$"
    latex += "\n"
    latex += r"$+ \sum\limits^{S^2}_i \mathbbm{1}_i^{\rm obj} \sum\limits_c (p_i(c) - \hat{p}_i(c))^2$"


    plt.figure(figsize=(9,4))
    plt.text(0, 0.1, latex, fontsize=20)                                  
    ax = plt.gca()
    ax.axis('off')
    plt.show()

    plt.figure
    im = plt.imread("papers/yolo_example.png")
    plt.imshow(im)
    ax = plt.gca()
    ax.axis('off')
    plt.show()


def check_calculated_squared_error():
    """Check that the squared error is being created correctly
    """

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
                test_yolo.LossFnTests
            )
    unittest.TextTestRunner().run(suite)


def check_yolo_as_classifier():
    """Rebuild the loss function so that it only rates the network on it's 
    classification abilities
    """
    # import data from binary file
    yolo_trainer = YOLOTrainer("./data/images.npy", "./data/labels.npy")

    # Initialize the network, show that increasing learning rate doesn't 
    # stabilize system 
    tinyyolo = YOLO(
            config_file="network_configs/tiny_yolo.json", 
            debug=True,
            learning_rate=1e-2
            )

    perr = tinyyolo.debug_dump["perr"]
    indicator_obj_i = tinyyolo.debug_dump["indicator_obj_i"]

    # make loss only the squared error of the class prediction 
    loss = tf.reduce_mean(
            indicator_obj_i * tf.reduce_sum(
                perr,
                axis=3,
                keep_dims=True
                )
            )

    tinyyolo.loss = loss


    # train
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tinyyolo.set_session(session)
        tinyyolo.train(yolo_trainer, batch_sz=1)


def check_input_data():
    """Run some basic tests on the data used
    """
    # load data from pickled files
    yolo_trainer = YOLOTrainer("./data/images.npy", "./data/labels.npy")
    # run tests on imported data
    yolo_trainer.test_self()


def run_yolo_with_debugger():
    """Run the network with the tensorflow debugger
    """
    # import data from binary file
    yolo_trainer = YOLOTrainer("./data/images.npy", "./data/labels.npy")

    # Initialize the network, show that increasing learning rate doesn't 
    # stabilize system 
    # learning_rate=1e-2
    # learning_rate=1e-8
    tinyyolo = YOLO(
            config_file="network_configs/tiny_yolo.json", 
            debug=True,
            learning_rate=1e-8
            )

    # train
    with tf.Session() as session:
        session = tf_debug.LocalCLIDebugWrapperSession(session)
        session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        session.run(tf.global_variables_initializer())
        tinyyolo.set_session(session)
        tinyyolo.train(yolo_trainer, batch_sz=1)


if __name__ == "__main__": 
    ## Show the data being used
    # visulize_yolo_input_data()

    ## Run yolo network to demonstrate non-convergence in loss
    # run_yolo_network()

    ## Display the loss equation 
    generate_latex_loss_eqn()

    ## Check that the calculated squared error is correct
    # check_calculated_squared_error()

