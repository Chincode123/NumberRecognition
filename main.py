import numpy as np
from abc import ABC, abstractmethod
import gzip
import pickle

class Activation(ABC):
    @abstractmethod
    def execute(self, z):
        pass

    @abstractmethod
    def derivative(self, z):
        pass

class Sigmoid(Activation):
    def execute(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):
        return self.execute(z) * (1 - self.execute(z))
    
class ReLU(Activation):
    def execute(self, z):
        return np.maximum(0, z)
    
    def derivative(self, z):
        return np.where(z > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, layers: list[int], activation: Activation = Sigmoid):
        self.layers = layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.activation = activation()
    
    def feedforward(self, inputs):
        for w, b in zip(self.weights, self.biases):
            inputs = self.activation.execute(np.dot(w, inputs) + b)
        return inputs
    
    def stocastic_gardient_decent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
                continue
            print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation.execute(z)
            activations.append(activation)
        delta = self.cost_derivitive(activations[-1], y) * self.activation.derivative(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.activation.derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w
    
    def cost_derivitive(self, output_activations, y):
        return (output_activations - y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def load_mnist_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    
    return training_data, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    network = NeuralNetwork([784, 40, 20, 10], Sigmoid)
    training_data, test_data = load_mnist_data()
    print("Starting training...")
    network.stocastic_gardient_decent(training_data, epochs=30, mini_batch_size=10, learning_rate=3.0, test_data=test_data)