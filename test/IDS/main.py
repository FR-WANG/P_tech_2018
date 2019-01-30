from __future__ import division, print_function, absolute_import
from NetworkBuilderV3 import NetworkBuilder
import numpy as np
import os
import tensorflow as tf
import gen_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
import visualization

# Print everything
np.set_printoptions(threshold=np.inf)

# Distinguish the function of train and test
is_train = False
# choose if cross-validation is done during the training
validation = True

# Definition the number of epochs and batchSize
epochs = 50
batchSize = 50

# Definition of the path : to change accordingly to your path leading to
# the model
model_save_path = "/comptes/etudiant/E17C429K/PycharmProjects/modele/"
model_name = 'Modele1'


# Main function


def main():
    # Load data and change the label to number
    train, test, crossval = gen_data.loaddata()
    train, test, crossval = gen_data.changelabel(train, test, crossval)

    print(type(train))

    # Separate the data into three sets and change the labels to one-hot encoding
    x_train = train.iloc[:, 1:79]
    y_train = train.iloc[:, 79]
    x_test = test.iloc[:, 1:79]
    y_test = test.iloc[:, 79]
    x_crossval = crossval.iloc[:, 1:79]
    y_crossval = crossval.iloc[:, 79]
    y_train = gen_data.one_hot_coding(y_train, 15)
    y_test = gen_data.one_hot_coding(y_test, 15)
    y_crossval = gen_data.one_hot_coding(y_crossval, 15)

    # Normalization
    x_train, x_test, x_crossval = gen_data.normalize(
        x_train, x_test, x_crossval)

    print(
        x_train.shape,
        y_train.shape,
        x_test.shape,
        y_test.shape,
        x_crossval.shape,
        y_crossval.shape)

    # Create the neural Network
    input_data = tf.placeholder(
        dtype='float', shape=[None, 78], name='input')
    target_labels = tf.placeholder(
        dtype='float', shape=[None, 15], name='target')
    nb = NetworkBuilder("Reseau1", input_data, 3, [256, 256, 15], 4, 1)
    nb.create_network()

    # Definition of the optimizer with Tensorflow
    with tf.name_scope("Optimization") as scope:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=nb.model, labels=target_labels)
        cost = tf.reduce_mean(cost)
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(
            cost, global_step=global_step)

    # Definition of the accuracy with Tensorflow
    with tf.name_scope('accuracy') as scope:
        correct_pred = tf.equal(
            tf.argmax(
                nb.prediction, 1), tf.argmax(
                target_labels, 1))
        accuracy = tf.reduce_mean(
            tf.cast(
                correct_pred,
                tf.float32),
            name='acu')
        tf.summary.scalar("acu", accuracy)

    # Saver for the model
    saver = tf.train.Saver()

    # Main process of tensorflow
    with tf.Session() as sess:

        if is_train:

            summary_merged = tf.summary.merge_all()
            filename = "./summary_log/run1"
            # Setting global steps

            tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
            tf.global_variables_initializer().run()

            if os.path.exists(model_save_path +
                              'checkpoint'):
                # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
                saver.restore(
                    sess, tf.train.latest_checkpoint(model_save_path))
            writer = tf.summary.FileWriter(filename, sess.graph)

            for epoch in range(epochs):

                data, label, batch_num = gen_data.shuffle(x_train,
                                                          y_train, batchSize)
                steps = 0

                for i in range(batch_num):

                    data_batch, label_batch = gen_data.batch(
                        data, label, i, batchSize)

                    error, sum_out, acu, steps, _ = sess.run([cost, summary_merged, accuracy, global_step, optimizer],
                                                             feed_dict={input_data: data_batch,
                                                                        target_labels: label_batch})
                    writer.add_summary(sum_out, steps)
                    if i % 1000 == 0:
                        print(
                            "epoch=",
                            epoch,
                            "Total Samples Trained=",
                            steps *
                            batchSize,
                            "err=",
                            error,
                            "accuracy=",
                            acu)
                # Save the model every epoch
                if epoch % 1 == 0:
                    print("Saving the model")
                    saver.save(
                        sess,
                        model_save_path +
                        model_name,
                        global_step=steps)
                # Loss and Accuracy on train and cross-validation set every 10
                # epochs
                if (epoch % 10 == 0) and validation:
                    cross_loss, cross_acc = sess.run([cost, accuracy], feed_dict={
                        input_data: x_crossval, target_labels: y_crossval})
                    train_loss, train_acc = sess.run([cost, accuracy], feed_dict={
                        input_data: x_crossval, target_labels: y_crossval})
                    print(
                        'Train  ---   Loss :%f, Accuracy :%f' %
                        (train_loss, train_acc))
                    print(
                        'Cross-validation  ---   Loss :%f, Accuracy :%f' %
                        (cross_loss, cross_acc))

            # Loss and Accuracy on train and cross-validation set in the end of
            # training
            cross_loss, cross_acc, cross_prediction = sess.run([cost, accuracy, nb.prediction], feed_dict={
                input_data: x_crossval, target_labels: y_crossval})
            train_loss, train_acc, train_prediction = sess.run([cost, accuracy, nb.prediction], feed_dict={
                input_data: x_train, target_labels: y_train})

            # Confusion matrix of training set
            train_true = np.argmax(y_train, axis=1)
            train_predict = np.argmax(train_prediction, axis=1)
            train_confuse_mat = sess.run(
                tf.convert_to_tensor(
                    tf.confusion_matrix(
                        train_true, train_predict)))
            print(train_confuse_mat)

            # Confusion matrix of cross validation set
            cross_true = np.argmax(y_crossval, axis=1)
            cross_predict = np.argmax(train_prediction, axis=1)
            confuse_mat = sess.run(
                tf.convert_to_tensor(
                    tf.confusion_matrix(
                        cross_true, cross_predict)))
            print(confuse_mat)
            visualization.plot_confusion_matrix(confuse_mat)

            # Accuracy and recall
            train_precision = precision_score(
                train_true, train_predict, average=None)
            train_recall = recall_score(
                train_true, train_predict, average=None)
            cross_precision = precision_score(
                cross_true, cross_predict, average=None)
            cross_recall = recall_score(
                cross_true, cross_predict, average=None)

            # Print the result
            print(
                'Train  ---   Loss :%f, Accuracy :%f, precision*recall :%f' %
                (train_loss, train_acc, train_precision * train_recall))
            print(
                'Cross-validation  ---   Loss :%f, Accuracy :%f, precision*recall :%f' %
                (cross_loss, cross_acc, cross_precision * cross_recall))

        else:
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
            test_loss, test_acc, test_prediction = sess.run([cost, accuracy, nb.prediction], feed_dict={
                input_data: x_test, target_labels: y_test})
            print('val_loss:%f, val_acc:%f' % (test_loss, test_acc))

            # Confusion matrix
            test_true = np.argmax(y_test, axis=1)
            test_predict = np.argmax(test_prediction, axis=1)
            confuse_mat = sess.run(
                tf.convert_to_tensor(
                    tf.confusion_matrix(
                        test_true, test_predict)))
            print(confuse_mat)
            visualization.plot_confusion_matrix(confuse_mat)


if __name__ == "__main__":
    main()
