import tensorflow as tf

class NetworkBuilder:
    def __init__(self):
        pass

    def attach_dense_relu_layer(self, input_layer, size):
        with tf.name_scope("DenseRelu") as scope:
            input_size = input_layer.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal([input_size, size]), name='dense_weigh')
            biases = tf.Variable(tf.random_normal([size]), name='dense_biases')
            return tf.nn.relu_layer(input_layer,weights,biases)
        
    def attach_relu_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            return tf.nn.relu(input_layer)

    def attach_sigmoid_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            return tf.nn.sigmoid(input_layer)

    def attach_softmax_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            return tf.nn.softmax(input_layer)

    def attach_dense_layer(self, input_layer, size, summary=False):
        with tf.name_scope("Dense") as scope:
            input_size = input_layer.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal([input_size, size]), name='dense_weigh')
            if summary:
                tf.summary.histogram(weights.name, weights)
            biases = tf.Variable(tf.random_normal([size]), name='dense_biases')
            dense = tf.matmul(input_layer, weights) + biases
            return dense, weights, biases
