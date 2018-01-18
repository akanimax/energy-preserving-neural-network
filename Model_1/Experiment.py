""" The Script for running the Energy-Delta-Regularized network models
    author: Animesh Karnewar
    Date: 18/01/2018
"""

import tensorflow as tf
import os
import Energy_Delta_Regularized_Network as NN

flags = tf.app.flags
FLAGS = flags.FLAGS
base_data_path = "../Data"
base_model_path = "Models"
mnist_data_path = os.path.join(base_data_path, "MNIST_data")


def main(_):
    """
    Main method used for running the tensorflow application
    :param _: Not used (tf passes the sys.argv by default)
    :return: Nothing
    """
    # get the data to work with
    train_x, train_y, dev_x, dev_y, test_x, test_y = NN.DataSetup.setup_mnist_data(mnist_data_path)

    # print the Data Information
    print("\n=============================================================")
    print("MNIST Data Information")
    print("=============================================================")
    print("Training Data Images and Labels:", train_x.shape, train_y.shape)
    print("Dev Data Images and Labels:", dev_x.shape, dev_y.shape)
    print("Test Data Images and Labels:", test_x.shape, test_y.shape)
    print("=============================================================")

    # setup some constants for the data
    no_training_examples, img_dim, n_channels = train_x.shape[0], train_x.shape[1], train_y.shape[1]

    print(no_training_examples, img_dim, n_channels)


if __name__ == '__main__':
    # TODO Setup the Arguments required for this.

    # Start the tensorflow application
    tf.app.run(main)
