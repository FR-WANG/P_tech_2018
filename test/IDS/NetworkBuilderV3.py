import tensorflow as tf


class NetworkBuilder:
    '''Classe permettant la création automatisée du réseau de neurones
    Paramètres :
    - name : permet de donner un nom au réseau
    - inputSize : nombre de paramètres d'entrée
    - outputSize : nombre de classes en sortie
    - nbLayers : nombre de couches du réseau de neurones
    - sizeLayers : tableau contenant le nombre de neurones pour chaque couche
    - activation : permet de choisir la fonction d'activation utilisée entre chaque couche
        1 - Relu
        2 - Leaky Relu
        3 - Elu
    - drop : taux de drop entre les couches (entre 0 et 1)'''

    def __init__(
            self,
            name,
            inputLayer,
            nbLayers,
            sizeLayers,
            activation,
            drop):
        self.name = name
        self.input_data = inputLayer
        self.nbLayers = nbLayers
        self.sizeLayers = sizeLayers
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
                if self.activationType == 2:
                    self.model = tf.nn.leaky_relu(self.model)
                if self.activationType == 3:
                    self.model = tf.nn.elu(self.model)

                if self.droprate != 0:
                    self.model = tf.layers.dropout(self.model, self.droprate)
