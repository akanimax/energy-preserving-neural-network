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
    total_train_examples, img_dim, n_classes = train_x.shape[0], train_x.shape[1], train_y.shape[1]

    # create the computation graph
    cmp_graph, int_dict = NN.NetworkBuilder.mk_graph(img_dim, n_classes, FLAGS.hid_repr_size, FLAGS.depth)

    # define the Training and Other additional ops
    with cmp_graph.as_default():
        # extract the tensors from the interface dictionary
        output = int_dict["output"]
        labels = int_dict["labels"]
        tf_input_images = int_dict["input"]
        energy_delta = int_dict["energy_delta"]

        # define the predictions
        with tf.name_scope("Predictions"):
            predictions = tf.nn.softmax(output)

        # define the loss function
        with tf.name_scope("Loss"):
            ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))

            if FLAGS.energy_delta_regularizer:
                regularization = FLAGS.regularization_lambda * energy_delta
                loss = ce + regularization

                tf.summary.scalar("Cross_Entropy_Loss", ce)
                tf.summary.scalar("Regularization_Loss", regularization)
            else:
                loss = ce

            # add scalar summary on this:
            tf.summary.scalar("Total_Loss", loss)

        # define the Trainer
        with tf.name_scope("Trainer"):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_step = optimizer.minimize(loss)

        # define the accuracy
        with tf.name_scope("Accuracy"):
            correct = tf.equal(tf.argmax(predictions, axis=-1), tf.argmax(labels, axis=-1))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.cast(tf.shape(labels)[0], tf.float32)

            tf.summary.scalar("Accuracy", accuracy)

        # finally define the required errands:
        with tf.name_scope("Errands"):
            init = tf.global_variables_initializer()
            all_sums = tf.summary.merge_all()

    # Infer the Model_name_from the configuration
    model_name = (str(FLAGS.depth) + "-deep-" +
                  str(FLAGS.hid_repr_size) + "-hdr-" +
                  str(FLAGS.epochs) + "-epochs-" +
                  str(FLAGS.energy_delta_regularizer) + "-energy_regularization-" +
                  str(FLAGS.learning_rate) + "-learning_rate-" +
                  str(FLAGS.regularization_lambda) + "-regularization_lambda-" +
                  str(FLAGS.batch_size) + "-batch_size")

    # Finally, define the training session:
    with tf.Session(graph=cmp_graph) as sess:
        # create a tensorboard writer
        model_save_path = os.path.join(base_model_path, model_name)
        tensorboard_writer = tf.summary.FileWriter(logdir=model_save_path, graph=sess.graph, filename_suffix=".bot")

        # create a saver
        saver = tf.train.Saver(max_to_keep=2)

        # restore the session if the checkpoint exists:
        if os.path.isfile(os.path.join(model_save_path, "checkpoint")):
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

        else:  # initialize all the variables:
            sess.run(init)

        global_step = 0
        no_of_epochs = FLAGS.epochs
        training_batch_size = FLAGS.batch_size
        print("Starting the training process . . .")
        for epoch in range(no_of_epochs):
            # run through the batches of the data:
            accuracies = []  # initialize this to an empty list
            runs = int((total_train_examples / training_batch_size) + 0.5)
            checkpoint = runs / 10
            for batch in range(runs):
                start = batch * training_batch_size
                end = start + training_batch_size

                # extract the relevant data:
                batch_data_x = train_x[start: end]
                batch_data_y = train_y[start: end]

                # This is batch gradient descent: (We are running it only on first 512 images)
                _, cost, acc, sums = sess.run([train_step, loss, accuracy, all_sums],
                                              feed_dict={tf_input_images: batch_data_x,
                                                         labels: batch_data_y})

                # append the acc to the accuracies list
                accuracies.append(acc)

                # save the summaries
                if batch % checkpoint == 0:
                    tensorboard_writer.add_summary(sums, global_step)

                # increment the global step
                global_step += 1

            print("\nepoch = ", epoch, "cost = ", cost)

            # evaluate the accuracy of the whole dataset:
            print("accuracy = ", sum(accuracies) / len(accuracies))
            # evaluate the accuracy for the dev set
            dev_acc = sess.run(accuracy, feed_dict={tf_input_images: dev_x, labels: dev_y})
            print("dev_accuracy = ", dev_acc)

            # save the model after every epoch
            saver.save(sess, os.path.join(model_save_path, model_name), global_step=(epoch + 10))

        # Once, the training is complete:
        # print the test accuracy:
        acc = sess.run(accuracy, feed_dict={tf_input_images: test_x, labels: test_y})
        print("Training complete . . .")
        print("Obtained Test accuracy = ", acc)


if __name__ == '__main__':
    # Define the Command line flags for the script
    flags.DEFINE_integer("hid_repr_size", 512, "The hidden representation size for the layers")

    flags.DEFINE_integer("depth", 1, "The depth of the Neural Network")

    flags.DEFINE_boolean("energy_delta_regularizer", True,
                         "Boolean switch for turning on and off the regularization")

    flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for training")

    flags.DEFINE_integer("batch_size", 64, "Batch size for training the network")

    flags.DEFINE_integer("epochs", 12, "No of epochs for which the data needs to be trained")

    flags.DEFINE_float("regularization_lambda", 0.1, "Used if --energy_delta_regularizer is set")

    # Start the tensorflow application
    tf.app.run(main)
