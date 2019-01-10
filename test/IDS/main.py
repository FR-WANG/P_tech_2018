from __future__ import division, print_function, absolute_import
import NetworkBuilder
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


def main():
    train, test, crossval = gen_data.loaddata()
    train, test, crossval = gen_data.changelabel(train, test, crossval)

    # normalization

    train, train_mean, train_std = gen_data.z_normalisation(train)
    test,test_mean,test_std = gen_data.z_normalisation(test)
    crossval,cross_mean,cross_std = gen_data.z_normalisation(crossval)
    X_TRAIN = train[:, 1:79]
    Y_TRAIN = train[:, 79]
    X_TEST = test[:, 1:79]
    Y_TEST = test[:, 79]
    X_CROSSVAL = crossval[:, 1:79]
    Y_CROSSVAL = crossval[:, 79]

    print(
        X_TRAIN.shape,
        Y_TRAIN.shape,
        X_TEST.shape,
        Y_TEST.shape,
        X_CROSSVAL.shape,
        Y_CROSSVAL.shape)


if __name__ == "__main__":
    main()

'''
with tf.name_scope("Input") as scope:
    input_data = tf.placeholder(dtype='float', shape=[None, 79], name='input')

with tf.name_scope("Target") as scope:
    target_labels = tf.placeholder(
        dtype='float', shape=[
            None, 15], name='target')


nb = NetworkBuilder()

with tf.name_scope("Modele") as scope:
    model = input_data
    model = nb.attach_dense_relu_layer(model, 256)
    model = nb.attach_dense_relu_layer(model, 256)
    model = nb.attach_dense_layer(model, 15)
    prediction = nb.attach_softmax_layer(model)


with tf.name_scope("Optimization") as scope:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=model, labels=target_labels)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost", cost)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.01).minimize(
        cost, global_step=global_step)

with tf.name_scope('accuracy') as scope:
    correct_pred = tf.equal(
        tf.argmax(
            prediction, 1), tf.argmax(
            target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    summaryMerged = tf.summary.merge_all()
    filename = "./summary_log"
    writer = tf.summary.FileWriter(filename, sess.graph)

epochs = 300
batchSize = 1

saver = tf.train.Saver()
model_save_path = "./modele/"
model_name = 'Modele1'

with tf.Session() as sess:
    summaryMerged = tf.summary.merge_all()

    filename = "./summary_log/run1"
    # setting global steps
    tf.global_variables_initializer().run()

    if os.path.exists(model_save_path + 'checkpoint'):
        # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
    writer = tf.summary.FileWriter(filename, sess.graph)

    for epoch in range(epochs):

        data_batch, label_batch = shuffle_and_batch(data, labels, batchSize)

        error, sumOut, acu, steps, _ = sess.run([cost, summaryMerged, accuracy, global_step, optimizer],
                                                feed_dict={input_data: data_batch, target_labels: label_batch})
        writer.add_summary(sumOut, steps)
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
        if steps % 100 == 0:
            print("Saving the model")
            saver.save(sess, model_save_path + model_name, global_step=steps)


input("Prompt")
'''
