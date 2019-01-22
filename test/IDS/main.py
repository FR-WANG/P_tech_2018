from __future__ import division, print_function, absolute_import
from NetworkBuilderV2 import NetworkBuilder
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import walk
from shutil import copyfile
import os
import pickle
import tensorflow as tf
import csv
import gen_data

#print everything
np.set_printoptions(threshold=np.inf)


# distinguish the fonction of train and test
is_train = True
#choose if cross-validation is done during the training
validation = True


# definition the number of epochs and batchSize
epochs = 50
batchSize = 50

# definition of the path : to change accordingly to your path leading to the model
model_save_path = "/comptes/etudiant/E17C429K/PycharmProjects/modele/"
model_name = 'Modele1'

# main fonction


def main():

    # load data and change the label to number
    train, test, crossval = gen_data.loaddata()
    train, test, crossval = gen_data.changelabel(train, test, crossval)

    print(type(train))

    # diviser the data to three set and change them to forme onhot
    X_TRAIN = train.iloc[:, 1:79]
    Y_TRAIN = train.iloc[:, 79]
    X_TEST = test.iloc[:, 1:79]
    Y_TEST = test.iloc[:, 79]
    X_CROSSVAL = crossval.iloc[:, 1:79]
    Y_CROSSVAL = crossval.iloc[:, 79]
    Y_TRAIN = gen_data.one_hot_coding(Y_TRAIN, 15)
    Y_TEST = gen_data.one_hot_coding(Y_TEST, 15)
    Y_CROSSVAL = gen_data.one_hot_coding(Y_CROSSVAL, 15)

    # normalization

    X_TRAIN, train_mean, train_std = gen_data.z_normalisation(X_TRAIN)
    X_TEST, test_mean, test_std = gen_data.z_normalisation(X_TEST)
    X_CROSSVAL, cross_mean, cross_std = gen_data.z_normalisation(X_CROSSVAL)
    print(
        X_TRAIN.shape,
        Y_TRAIN.shape,
        X_TEST.shape,
        Y_TEST.shape,
        X_CROSSVAL.shape,
        Y_CROSSVAL.shape)

    # create the neural Network
    input_data = tf.placeholder(
        dtype='float', shape=[None, 78], name='input')
    target_labels = tf.placeholder(
        dtype='float', shape=[None, 15], name='target')
    nb = NetworkBuilder("Reseau1", input_data, 3, [256, 256, 15])
    nb.create_network()

    # definition of the optimizer with tensorflow
    with tf.name_scope("Optimization") as scope:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=nb.model, labels=target_labels)
        cost = tf.reduce_mean(cost)
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(
            cost, global_step=global_step)

    # definition of the accuracy with tensorflow
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

    # summary for tensorboard
    with tf.Session() as sess:
        summaryMerged = tf.summary.merge_all()
        filename = "./summary_log"
        writer = tf.summary.FileWriter(filename, sess.graph)

    # saver for save the model
    saver = tf.train.Saver()

    # main process of tensorflow
    with tf.Session() as sess:

        if is_train:
            summaryMerged = tf.summary.merge_all()
            filename = "./summary_log/run1"
            # setting global steps
            tf.global_variables_initializer().run()

            if os.path.exists(model_save_path + 'checkpoint'):
                # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
                saver.restore(
                    sess, tf.train.latest_checkpoint(model_save_path))
            writer = tf.summary.FileWriter(filename, sess.graph)

            for epoch in range(epochs):

                data, label, batch_num = gen_data.shuffle(
                    X_TRAIN, Y_TRAIN, batchSize)

                for i in range(batch_num):

                    data_batch, label_batch = gen_data.batch(
                        data, label, i, batchSize)

                    error, sumOut, acu, steps, _ = sess.run([cost, summaryMerged, accuracy, global_step, optimizer], feed_dict={
                                                            input_data: data_batch, target_labels: label_batch})
                    writer.add_summary(sumOut, steps)
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
                #Save the model every epoch
                if epoch % 1 == 0:
                    print("Saving the model")
                    saver.save(
                        sess,
                        model_save_path +
                        model_name,
                        global_step=steps)
                #Accuracy on cross-validation set every 10 epochs
                if (epoch % 10 == 0) and (validation == True):
                    cross_loss, cross_acc = sess.run([cost, accuracy], feed_dict={
                        input_data: X_CROSSVAL, target_labels: Y_CROSSVAL})
                    print('Cross-validation  ---   Loss :%f, Accuracy :%f' % (cross_loss, cross_acc))
        else:
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
            test_loss, test_acc, pred = sess.run([cost, accuracy, nb.prediction], feed_dict={
                input_data: X_TEST, target_labels: Y_TEST})
            print('val_loss:%f, val_acc:%f' % (test_loss, test_acc))

            # confusion matrix
            Y_true = np.argmax(Y_TEST, axis=1)

            Y_predict = np.argmax(pred, axis=1)
            confuse_martix = sess.run(
                tf.convert_to_tensor(
                    tf.confusion_matrix(
                        Y_true, Y_predict)))
            print(confuse_martix)


if __name__ == "__main__":
    main()
