import tensorflow as tf


class NetworkBuilder:
    '''Classe permettant la création automatisée du réseau de neurones
    Paramètres :
    - name : permet de donner un nom au réseau
    - inputSize : nombre de paramètres d'entrée
    - outputSize : nombre de classes en sortie
    - nbLayers : nombre de couches du réseau de neurones
    - sizeLayers : tableau contenant le nombre de neurones pour chaque couche'''

    def __init__(self, name, inputSize, outputSize, nbLayers, sizeLayers):
        self.name = name
        self.input_data = tf.placeholder(
            dtype='float', shape=[None, inputSize], name='input')
        self.target_labels = tf.placeholder(
            dtype='float', shape=[None, outputSize], name='target')
        self.nbLayers = nbLayers
        self.sizeLayers = sizeLayers
        self.model = self.input_data

    def create_network(self):
        for i in range(0, self.nbLayers - 1):
            input_size = self.model.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal(
                [input_size, self.sizeLayers[i]]), name='dense_weigh')
            biases = tf.Variable(tf.random_normal(
                [self.sizeLayers[i]]), name='dense_biases')
            self.model = tf.matmul(self.model, weights) + biases
            if i == self.nbLayers - 1:
                self.model = tf.nn.softmax(self.model)
            else:
                self.model = tf.nn.relu(self.model)
