""" The Network Builder has only functional interface.

    ** This is a preliminary implementation.
    ** Refactoring is required
"""
import tensorflow as tf


def mk_graph(img_dim, num_labels, hidd_repr_size=512, depth=1):
    """
    The function that creates and returns The Computation graph of the Neural Network
    :param img_dim: The dimensionality of the input images
    :param num_labels: The number of classes for classification
    :param hidd_repr_size: The size of the hidden representations of the Neural Network
    :param depth: The depth of the Neural Network
    :return: The tensorflow computation Graph and the interface dictionary
    """

    # static graph building process. This builds the graph for the given configuration
    comp_graph = tf.Graph()

    with comp_graph.as_default():
        # step 1: Create the input placeholders for the input to the computation
        with tf.name_scope("Input"):
            tf_input_images = tf.placeholder(tf.float32, shape=(None, img_dim), name="Input_Labels")
            tf_input_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name="Input_Labels")
        print("\nInput Placeholder Tensors:", tf_input_images, tf_input_labels)

        # step 2: Create the Neural Network Architecture
        if depth > 1:
            # The default initializer is the GLOROT_uniform_initializer
            neural_layer = tf.layers.Dense(hidd_repr_size, activation=tf.nn.selu, name="Layer1")

            # Note: This needs to be before summary attaching
            lay1_out = neural_layer(tf_input_images)

        else:
            neural_layer = tf.layers.Dense(num_labels, name="Layer1")
            lay1_out = neural_layer(tf_input_images)

        with tf.name_scope("Energy_conservation/Layer1"):
            energy_delta = tf.abs(tf.reduce_sum(tf.norm(tf_input_images, axis=-1))
                                  - tf.reduce_sum(tf.norm(lay1_out, axis=-1)))

        # attach histogram summaries to the trainable weights of this layer
        wt1, wt2 = neural_layer.trainable_weights
        tf.summary.histogram("Layer1/kernel", wt1)
        tf.summary.histogram("Layer1/bias", wt2)

        # Create the in-between layers (Hidden layers) of the neural network
        lay_out = lay1_out  # initialize to output of first layer
        for lay_no in range(2, depth):
            neural_layer = tf.layers.Dense(hidd_repr_size, activation=tf.nn.selu, name="Layer" + str(lay_no))
            prev_out = lay_out
            lay_out = neural_layer(lay_out)

            with tf.name_scope("Energy_conservation/Layer" + str(lay_no)):
                energy_delta = tf.add(energy_delta, (tf.abs(tf.reduce_sum(tf.norm(prev_out, axis=-1))
                                                            - tf.reduce_sum(tf.norm(lay_out, axis=-1)))))

            # attach the weight summaries
            wt1, wt2 = neural_layer.trainable_weights
            tf.summary.histogram("Layer" + str(lay_no) + "/kernel", wt1)
            tf.summary.histogram("Layer" + str(lay_no) + "/bias", wt2)

        # define the final output layer
        if depth > 1:
            neural_layer = tf.layers.Dense(num_labels, name="Layer" + str(depth))
            output = neural_layer(lay_out)

            wt1, wt2 = neural_layer.trainable_weights
            tf.summary.histogram("Layer" + str(depth) + "/kernel", wt1)
            tf.summary.histogram("Layer" + str(depth) + "/bias", wt2)
        else:
            output = lay1_out

        with tf.name_scope("Energy_conservation/Layer" + str(depth)):
            energy_delta = tf.add(energy_delta, (tf.abs(tf.reduce_sum(tf.norm(lay_out, axis=-1))
                                                        - tf.reduce_sum(tf.norm(output, axis=-1)))))

        print("Final output:", output)

    return comp_graph, {"output": output, "labels": tf_input_labels,
                        "input": tf_input_images, "energy_delta": energy_delta}
