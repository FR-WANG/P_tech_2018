import tensorflow as tf


class NetworkBuilder:
    """
    Class creating the neural network

    Parameter
    ---------

    name : str
        gives a name to the network
    inputLayer : Tensorflow placeholder
        input of the network
    nbLayers : int
        number of layers in the neural network
    sizeLayers : int array
        array defining the number of nodes for each layer
    activation : int
        allows to chose the activation function (default is Relu):
            1 - Relu
            2 - Leaky Relu
            3 - Elu
            4 - Selu
    drop : float
        probability to keep, between 0 and 1


    Return
    ------

    Neural network

    """

    def __init__(
            self,
            name,
            input_layer,
            nb_layers,
            size_layers,
            activation,
            drop):
        self.name = name
        self.input_data = input_layer
        self.nbLayers = nb_layers
        self.sizeLayers = size_layers
        self.model = self.input_data
        self.prediction = self.model
        self.droprate = drop
        self.activationType = activation

    def create_network(self):
        for i in range(0, self.nbLayers):
            input_size = self.model.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal(
                [input_size, self.sizeLayers[i]]), name='dense_weigh')
            biases = tf.Variable(tf.random_normal(
                [self.sizeLayers[i]]), name='dense_biases')
            self.model = tf.matmul(self.model, weights) + biases
            if i == self.nbLayers - 1:
                self.prediction = tf.nn.softmax(self.model)
            else:
                if self.activationType == 1:
                    self.model = tf.nn.relu(self.model)
                elif self.activationType == 2:
                    self.model = tf.nn.leaky_relu(self.model)
                elif self.activationType == 3:
                    self.model = tf.nn.elu(self.model)
                elif self.activationType == 4:
                    self.model = tf.nn.selu(self.model)
                else:
                    self.model = tf.nn.relu(self.model)

                if self.droprate != 1:
                    self.model = tf.layers.dropout(self.model, self.droprate)
