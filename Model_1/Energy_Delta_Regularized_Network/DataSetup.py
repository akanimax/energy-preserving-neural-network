""" Function for setting up the MNIST data
"""
from tensorflow.examples.tutorials.mnist import input_data


def setup_mnist_data(data_dir, one_hot=True, normalized=True):
    """ The function for setting up the mnist data
    :param data_dir: String indicating the path of the data directory
    :param one_hot: Boolean for indicating if the labels are required in one-hot encoded format
    :param normalized: Boolean for indicating if the data needs to be normalized
    :return: tuple containing np arrays of the split data
    """
    # obtain the data from the data directory
    print("\nObtaining Data ...")
    mnist_data = input_data.read_data_sets(data_dir, one_hot=one_hot)

    # divide the data into train, dev and test sets
    train_x = mnist_data.train.images; train_y = mnist_data.train.labels
    dev_x = mnist_data.validation.images; dev_y = mnist_data.validation.labels
    test_x = mnist_data.test.images; test_y = mnist_data.test.labels

    # if normalization is false scale up the data values
    if(not normalized):
        highest_pixel_value = 255
        train_x *= highest_pixel_value
        dev_x *= highest_pixel_value
        test_x *= highest_pixel_value

    # return a tuple of the formatted data
    return (train_x, train_y, dev_x, dev_y, test_x, test_y)